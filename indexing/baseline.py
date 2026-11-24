# =============================================================================
# indexing/baseline.py
# 效能比較的 baseline：Part 1 的基本建法，不限制記憶體
#
#   逐篇讀文件、做同一套前處理，term → postings list 全部放在記憶體（同一篇只記一次），
#   讀完所有文件後排序詞彙，一次寫出最終索引（和 BSBI、SPIMI 相同的格式，大小才能比）
#   沒有 blocks、沒有中間檔、不需要合併；代價是整份索引都要放得進記憶體，文件越多用得越多
#   （等於「記憶體無限大的 SPIMI」：只有 1 個 block，而且不必先寫成 block 再合併）
# =============================================================================
import shutil
import sys
from pathlib import Path
from time import perf_counter

from .collection import preprocess
from .storage import FinalIndexWriter

STAGES = ('preprocess', 'pairs', 'sort', 'write', 'merge')


# 參數和 bsbi.build、spimi.build 一樣（memory_limit 不使用），回傳相同格式的結果
def build(documents, output, memory_limit=None, samples=False, trace_terms=()):
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    times = dict.fromkeys(STAGES, 0.0)
    index = {}                         # term → postings list（全部文件）
    documents_read = tokens = 0
    started = perf_counter()
    for doc_id, text in documents:
        clock = perf_counter()
        terms = preprocess(text)
        times['preprocess'] += perf_counter() - clock
        clock = perf_counter()
        for term in terms:
            postings = index.get(term)
            if postings is None:
                index[term] = [doc_id]
            elif postings[-1] != doc_id:
                postings.append(doc_id)
        times['pairs'] += perf_counter() - clock
        documents_read += 1
        tokens += len(terms)
    clock = perf_counter()
    terms = sorted(index)
    times['sort'] = perf_counter() - clock
    clock = perf_counter()
    with FinalIndexWriter(output / 'index') as writer:
        for term in terms:
            writer.add(term, index[term])
    times['write'] = perf_counter() - clock
    seconds = perf_counter() - started
    return {'method': 'baseline', 'seconds': seconds, 'times': times, 'blocks': [],
            'documents': documents_read, 'tokens': tokens, 'vocabulary': len(terms), 'term_id_map_bytes': 0,
            'index_memory': _index_bytes(index, documents_read), 'samples': None}


# 整份索引在記憶體裡的大小（sys.getsizeof 加總，不另外跑 tracemalloc 也能估）：
#   dictionary 本身 + 每個 term 字串 + 每個 postings list；docID 的整數物件每篇文件一個（28 bytes）
def _index_bytes(index, documents):
    return (sys.getsizeof(index) + sum(sys.getsizeof(term) + sys.getsizeof(postings) for term, postings in index.items())
            + 28 * documents)
