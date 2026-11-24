# =============================================================================
# indexing/storage.py
# Project 5：寫到磁碟的檔案格式（BSBI 與 SPIMI 共用），以及 partial indexes 的合併
#
# docID 一律存成 4 bytes 的無號整數（array 'I'，little-endian）。
#
# partial index（block 檔，blocks/block_001.bin …）：一筆一個 term，依 key 由小到大排好
#   BSBI ：termID（4 bytes）, df（4 bytes）, docID × df          ← key 是 termID
#   SPIMI：term 長度（2 bytes）, term（UTF-8）, df（4 bytes）, docID × df   ← key 是 term 字串
#
# 最終索引（兩種方法完全相同的格式，大小才能直接比較）：
#   dictionary.bin：一筆一個 term：term 長度（2 bytes）, term（UTF-8）, df（4 bytes）, offset（4 bytes）
#                   offset = 這個 term 的 postings 從 postings.bin 的第幾個 docID 開始
#   postings.bin  ：所有 postings lists 依序接在一起
#
# 合併（merge_blocks）：所有 block 檔同時打開、各讀第一筆，用 heap 每次取出 key 最小的；
#   不同 block 的同一個 key 依 block 順序接起來（block 本身就是依 docID 先後切的，接起來仍然排序好）。
#   記憶體裡同時只有「每個 block 目前這一筆」，所以合併也不需要把整個索引讀進記憶體。
# =============================================================================
import heapq
import struct
from array import array
from pathlib import Path

DOC_ID = 'I'
assert array(DOC_ID).itemsize == 4
_U16 = struct.Struct('<H')
_U32 = struct.Struct('<I')
_PAIR = struct.Struct('<II')
BUFFER = 1 << 16


def _docs(stream, count):
    postings = array(DOC_ID)
    postings.frombytes(stream.read(4 * count))
    return postings


def _read_term(stream):
    head = stream.read(2)
    if not head:
        return None
    return stream.read(_U16.unpack(head)[0]).decode('utf-8')


# ---------------------------------------------------------------------------
# partial index（block 檔）
# ---------------------------------------------------------------------------

def write_termid_record(stream, term_id, postings):
    stream.write(_PAIR.pack(term_id, len(postings)))
    stream.write(array(DOC_ID, postings).tobytes())


def write_term_record(stream, term, postings):
    raw = term.encode('utf-8')
    stream.write(_U16.pack(len(raw)) + raw + _U32.pack(len(postings)))
    stream.write(array(DOC_ID, postings).tobytes())


# 依序讀出 BSBI block 的 (termID, postings)
def read_termid_block(path):
    with open(path, 'rb', buffering=BUFFER) as stream:
        while head := stream.read(8):
            term_id, df = _PAIR.unpack(head)
            yield term_id, _docs(stream, df)


# 依序讀出 SPIMI block 的 (term, postings)
def read_term_block(path):
    with open(path, 'rb', buffering=BUFFER) as stream:
        while (term := _read_term(stream)) is not None:
            yield term, _docs(stream, _U32.unpack(stream.read(4))[0])


# k-way merge：readers 是每個 block 的 (key, postings) 產生器；依 key 由小到大產生合併後的 (key, postings)
def merge_blocks(readers):
    heap = []
    for number, reader in enumerate(readers):
        first = next(reader, None)
        if first is not None:
            heap.append((first[0], number, first[1]))
    heapq.heapify(heap)
    while heap:
        key, number, postings = heapq.heappop(heap)
        sources = [(number, postings)]
        while heap and heap[0][0] == key:
            _, other, more = heapq.heappop(heap)
            sources.append((other, more))
        for source, _ in sources:
            following = next(readers[source], None)
            if following is not None:
                heapq.heappush(heap, (following[0], source, following[1]))
        merged = sources[0][1]
        for _, more in sorted(sources[1:], key=lambda item: item[0]):
            merged.extend(more)
        yield key, merged


# ---------------------------------------------------------------------------
# 最終索引
# ---------------------------------------------------------------------------

class FinalIndexWriter:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.dictionary = open(self.folder / 'dictionary.bin', 'wb', buffering=BUFFER)
        self.postings = open(self.folder / 'postings.bin', 'wb', buffering=BUFFER)
        self.offset = self.terms = 0

    def add(self, term, postings):
        raw = term.encode('utf-8')
        self.dictionary.write(_U16.pack(len(raw)) + raw + _PAIR.pack(len(postings), self.offset))
        self.postings.write(postings.tobytes() if isinstance(postings, array) else array(DOC_ID, postings).tobytes())
        self.offset += len(postings)
        self.terms += 1

    def close(self):
        self.dictionary.close()
        self.postings.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# 讀最終索引的 dictionary：{term: (df, offset)}
def read_dictionary(folder):
    entries = {}
    with open(Path(folder) / 'dictionary.bin', 'rb', buffering=BUFFER) as stream:
        while (term := _read_term(stream)) is not None:
            entries[term] = _PAIR.unpack(stream.read(8))
    return entries


def index_size(folder):
    folder = Path(folder)
    return {'dictionary': (folder / 'dictionary.bin').stat().st_size, 'postings': (folder / 'postings.bin').stat().st_size}
