# =============================================================================
# indexing/collection.py
# Project 5：要建索引的文件（BSBI、SPIMI、Part 1 基準索引都從這裡讀，作業要求三者用同一批文件）
#
#   資料集（sources.py，網頁可以切換）：
#     reuters  Reuters-21578 全部 21,578 篇（標題 + 內文），沿用 Project 4 下載與解析好的資料；docID = NEWID（1–21578）
#     pubmed   PubMed GLP-1 20,000 篇（標題 + 摘要）；docID = 依 PMID 排序後的 1–20000
#   兩者都依 docID 由小到大依序讀取
#   前處理：和 Project 4 完全相同（reuters/text.py 的 preprocess）
#     1. Tokenization（HW1 的 tokenizer）
#     2. Lowercase normalization
#     3. Stop-word removal（HW1 的 72 個 stopword）
#     4. 移除沒有英文字母的 token（純數字、金額）
#     5. 不做 stemming / lemmatization（作業列為可選）
#
# 規模實驗（第 6 題要 10,000 → 100,000 → 1,000,000 篇）：語料只有兩萬多篇，
#   超過時把整個語料再讀一輪，docID 加上最後一個 docID 的倍數當成新文件。
#   重複的文件不會帶來新詞，所以詞彙量會被低估（結果頁有註明）。
# =============================================================================
from functools import lru_cache

from reuters.corpus import document_text, load_documents
from reuters.text import preprocess

from .sources import DEFAULT_SOURCE, pubmed_documents, pubmed_text

__all__ = ['corpus_size', 'preprocess', 'read_documents']


@lru_cache(maxsize=2)
def _documents(source=DEFAULT_SOURCE):
    if source == 'pubmed':
        documents = ((doc_id, pubmed_text(record)) for doc_id, record in pubmed_documents().items())
    else:
        documents = ((doc['id'], document_text(doc)) for doc in load_documents())
    return sorted(documents, key=lambda item: item[0])


def corpus_size(source=DEFAULT_SOURCE):
    return len(_documents(source))


# 依序產生 (docID, 原始文字)；count = None 代表整個語料，超過語料篇數就重複讀（docID 往後編號）
def read_documents(count=None, source=DEFAULT_SOURCE):
    documents = _documents(source)
    total = len(documents) if count is None else count
    span = documents[-1][0]
    produced = copy = 0
    while produced < total:
        for doc_id, text in documents:
            if produced == total:
                return
            yield copy * span + doc_id, text
            produced += 1
        copy += 1
