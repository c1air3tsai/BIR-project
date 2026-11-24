# =============================================================================
# indexing/bsbi.py
# Part 2：Blocked Sort-Based Indexing（BSBI）
#
#   Step 1  Divide documents into blocks：依序讀文件，把這個 block 的 (termID, docID) pairs 放在記憶體裡；
#           估計的記憶體用量到達上限（memory_limit）就結束這個 block（block 一定在兩篇文件之間切開）
#   Step 2  Generate term-document pairs：文件裡每個 token 都變成一個 (termID, docID)
#           termID 來自全域的 term → termID 對照表，所有 blocks 共用，所以同一個 term 在每個 block 的 termID 都一樣
#   Step 3  Sort：依 (termID, docID) 排序這個 block 的所有 pairs
#   Step 4  Build partial inverted index：排序後同一個 termID 的 pairs 連在一起，依序收集 docID
#           （同一篇文件的重複 pair 在這裡去掉），寫到磁碟成為 blocks/block_NNN.bin
#   Step 5  Merge：所有 block 檔同時打開，依 termID 做 k-way merge，寫出最終索引（storage.FinalIndexWriter）
#
# 記憶體上限的模擬：每個 pair 在 Python 裡是一個 tuple（sys.getsizeof((0, 0)) = 56 bytes）
#   再加上 list 裡的一個指標（8 bytes），所以 pairs 用量 = pairs 數 × 64 bytes。
#   全域的 term → termID 對照表一直留在記憶體（BSBI 本來就需要它），不算在 block 的上限裡，另外記錄它的大小。
# =============================================================================
import shutil
import sys
from pathlib import Path
from time import perf_counter

from .collection import preprocess
from .storage import FinalIndexWriter, merge_blocks, read_termid_block, write_termid_record, BUFFER

PAIR_BYTES = sys.getsizeof((0, 0)) + 8
STAGES = ('preprocess', 'pairs', 'sort', 'write', 'merge')


# documents：(docID, 文字) 依 docID 由小到大；output：這次建索引的資料夾（裡面放 blocks/ 與 index/）
# samples：要記下中間結果（第一個 block 的 pairs、排序後、局部索引，以及 trace_terms 在各 block 的 postings）
def build(documents, output, memory_limit, samples=False, trace_terms=()):
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    (output / 'blocks').mkdir(parents=True)
    times = dict.fromkeys(STAGES, 0.0)
    term_ids = {}                      # 全域 dictionary：term → termID（所有 blocks 共用）
    pairs = []                         # 目前 block 的 (termID, docID)
    blocks = []
    sample = {'pairs': [], 'sorted': [], 'partial': [], 'trace': {t: [] for t in trace_terms}} if samples else None
    first_doc = last_doc = None
    documents_read = tokens = 0
    started = perf_counter()

    def flush():
        nonlocal pairs, first_doc
        number = len(blocks) + 1
        if sample is not None and number == 1:
            sample['pairs'] = pairs[:16]
        clock = perf_counter()
        pairs.sort()                                                        # Step 3
        times['sort'] += perf_counter() - clock
        if sample is not None and number == 1:
            sample['sorted'] = pairs[:16]
        clock = perf_counter()
        path = output / 'blocks' / f'block_{number:03d}.bin'
        traced = {term_ids[t]: t for t in trace_terms if t in term_ids}
        terms = 0
        with open(path, 'wb', buffering=BUFFER) as stream:                 # Step 4
            current, postings = None, []
            for term_id, doc_id in pairs:
                if term_id != current:
                    if postings:
                        write_termid_record(stream, current, postings)
                        terms += 1
                        _remember(sample, number, current, postings, traced)
                    current, postings = term_id, [doc_id]
                elif postings[-1] != doc_id:
                    postings.append(doc_id)
            if postings:
                write_termid_record(stream, current, postings)
                terms += 1
                _remember(sample, number, current, postings, traced)
        times['write'] += perf_counter() - clock
        blocks.append({'block': number, 'first_doc': first_doc, 'last_doc': last_doc, 'pairs': len(pairs),
                       'terms': terms, 'bytes': path.stat().st_size, 'memory': len(pairs) * PAIR_BYTES})
        pairs = []
        first_doc = None

    for doc_id, text in documents:                                          # Step 1
        clock = perf_counter()
        terms = preprocess(text)
        times['preprocess'] += perf_counter() - clock
        clock = perf_counter()
        for term in terms:                                                  # Step 2
            term_id = term_ids.get(term)
            if term_id is None:
                term_id = term_ids[term] = len(term_ids)
            pairs.append((term_id, doc_id))
        times['pairs'] += perf_counter() - clock
        documents_read += 1
        tokens += len(terms)
        if first_doc is None:
            first_doc = doc_id
        last_doc = doc_id
        if len(pairs) * PAIR_BYTES >= memory_limit:
            flush()
    if pairs:
        flush()

    clock = perf_counter()                                                  # Step 5
    id_to_term = sorted(term_ids, key=term_ids.get)
    readers = [read_termid_block(output / 'blocks' / f"block_{b['block']:03d}.bin") for b in blocks]
    merged = []
    with FinalIndexWriter(output / 'index') as writer:
        for term_id, postings in merge_blocks(readers):
            writer.add(id_to_term[term_id], postings)
            if sample is not None and id_to_term[term_id] in sample['trace']:
                merged.append((id_to_term[term_id], term_id, list(postings)))
    times['merge'] = perf_counter() - clock
    if sample is not None:
        sample['merged'] = merged
        sample['partial'] = [(term_id, id_to_term[term_id], df, head) for term_id, df, head in sample['partial']]
        sample['pairs'] = [(term_id, id_to_term[term_id], doc_id) for term_id, doc_id in sample['pairs']]
        sample['sorted'] = [(term_id, id_to_term[term_id], doc_id) for term_id, doc_id in sample['sorted']]
        sample['term_ids'] = {term: term_ids[term] for term in trace_terms if term in term_ids}
    return {'method': 'bsbi', 'seconds': perf_counter() - started, 'times': times, 'blocks': blocks,
            'documents': documents_read, 'tokens': tokens, 'vocabulary': len(term_ids),
            'term_id_map_bytes': _dictionary_bytes(term_ids), 'samples': sample}


# 中間結果：第一個 block 的局部索引前幾筆、trace_terms 在每個 block 的 postings
def _remember(sample, number, term_id, postings, traced):
    if sample is None:
        return
    if number == 1 and len(sample['partial']) < 10:
        sample['partial'].append((term_id, len(postings), postings[:8]))
    if term_id in traced:
        sample['trace'][traced[term_id]].append((number, list(postings)))


# term → termID 對照表的大小估計：dict 本身 + 每個 term 字串（termID 是小整數物件，忽略）
def _dictionary_bytes(term_ids):
    return sys.getsizeof(term_ids) + sum(sys.getsizeof(term) for term in term_ids)
