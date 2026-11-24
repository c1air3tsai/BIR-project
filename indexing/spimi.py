# =============================================================================
# indexing/spimi.py
# Part 3：Single-Pass In-Memory Indexing（SPIMI）
#
#   Step 1  Read documents sequentially：逐篇讀文件、做前處理
#   Step 2  Build dictionary and postings lists：讀到一個 term 時
#             - term 還不在 dictionary → 建一個新的 entry（term → [docID]）
#             - term 已經在 dictionary → 把目前的 docID 加到它的 postings list
#             - 同一篇文件的同一個 term 只加一次（postings 最後一個就是這篇時跳過）
#           不產生 (termID, docID) pairs，也不需要 termID：dictionary 直接用 term 字串當 key
#   Step 3  Memory management：估計的記憶體用量到達上限（memory_limit）時，
#           把 dictionary 的 terms 排序後寫到磁碟（blocks/block_NNN.bin，一個 partial index），清空記憶體，繼續下一批文件
#           ——唯一的排序是「這個 block 的詞彙」，postings 本來就依讀取順序（docID 由小到大）排好
#   Step 4  Merge：所有 block 檔依 term 字串做 k-way merge，寫出最終索引（和 BSBI 完全相同的格式）
#
# 記憶體上限的模擬（同樣用 sys.getsizeof 量得的 CPython 物件大小）：
#   新 term：term 字串 + 只有一個 docID 的 list + dictionary 的一個 entry（約 32 bytes）
#   已有 term 的新 posting：list 裡多一個指標（8 bytes）
# =============================================================================
import shutil
import sys
from pathlib import Path
from time import perf_counter

from .collection import preprocess
from .storage import FinalIndexWriter, merge_blocks, read_term_block, write_term_record, BUFFER

ENTRY_BYTES = sys.getsizeof([0]) + 32
POSTING_BYTES = 8
STAGES = ('preprocess', 'pairs', 'sort', 'write', 'merge')


def build(documents, output, memory_limit, samples=False, trace_terms=()):
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    (output / 'blocks').mkdir(parents=True)
    times = dict.fromkeys(STAGES, 0.0)
    dictionary = {}                    # 目前 block 的 term → postings list
    used = 0                           # 目前 block 估計的記憶體用量（bytes）
    blocks = []
    sample = {'dictionary': [], 'sorted': [], 'trace': {t: [] for t in trace_terms}} if samples else None
    first_doc = last_doc = None
    documents_read = tokens = postings_count = 0
    started = perf_counter()

    def flush():
        nonlocal dictionary, used, first_doc, postings_count
        number = len(blocks) + 1
        if sample is not None and number == 1:
            sample['dictionary'] = [(term, len(p), p[:8]) for term, p in list(dictionary.items())[:10]]
        clock = perf_counter()
        terms = sorted(dictionary)                                          # 只排序詞彙
        times['sort'] += perf_counter() - clock
        if sample is not None and number == 1:
            sample['sorted'] = [(term, len(dictionary[term]), dictionary[term][:8]) for term in terms[:10]]
        clock = perf_counter()
        path = output / 'blocks' / f'block_{number:03d}.bin'
        with open(path, 'wb', buffering=BUFFER) as stream:
            for term in terms:
                write_term_record(stream, term, dictionary[term])
        times['write'] += perf_counter() - clock
        if sample is not None:
            for term in sample['trace']:
                if term in dictionary:
                    sample['trace'][term].append((number, list(dictionary[term])))
        blocks.append({'block': number, 'first_doc': first_doc, 'last_doc': last_doc, 'pairs': postings_count,
                       'terms': len(terms), 'bytes': path.stat().st_size, 'memory': used})
        dictionary, used, first_doc, postings_count = {}, 0, None, 0

    for doc_id, text in documents:                                          # Step 1
        clock = perf_counter()
        terms = preprocess(text)
        times['preprocess'] += perf_counter() - clock
        clock = perf_counter()
        for term in terms:                                                  # Step 2
            postings = dictionary.get(term)
            if postings is None:
                dictionary[term] = [doc_id]
                used += sys.getsizeof(term) + ENTRY_BYTES
                postings_count += 1
            elif postings[-1] != doc_id:
                postings.append(doc_id)
                used += POSTING_BYTES
                postings_count += 1
        times['pairs'] += perf_counter() - clock
        documents_read += 1
        tokens += len(terms)
        if first_doc is None:
            first_doc = doc_id
        last_doc = doc_id
        if used >= memory_limit:                                            # Step 3
            flush()
    if dictionary:
        flush()

    clock = perf_counter()                                                  # Step 4
    readers = [read_term_block(output / 'blocks' / f"block_{b['block']:03d}.bin") for b in blocks]
    merged, terms = [], 0
    with FinalIndexWriter(output / 'index') as writer:
        for term, postings in merge_blocks(readers):
            writer.add(term, postings)
            terms += 1
            if sample is not None and term in sample['trace']:
                merged.append((term, list(postings)))
    times['merge'] = perf_counter() - clock
    if sample is not None:
        sample['merged'] = merged
    return {'method': 'spimi', 'seconds': perf_counter() - started, 'times': times, 'blocks': blocks,
            'documents': documents_read, 'tokens': tokens, 'vocabulary': terms, 'term_id_map_bytes': 0,
            'samples': sample}
