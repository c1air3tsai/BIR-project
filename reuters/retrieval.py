# =============================================================================
# reuters/retrieval.py
# Project 4：新聞檢索引擎（檢索對象 = ModApte 測試集全部 3,299 篇）
#
# 方法 A：TF-IDF
#   文件 / 查詢向量的每一維是一個詞：w(t,d) = tf(t,d) × idf(t)，idf(t) = log10(N / df(t))
#   分數 = cosine(q, d)（向量先 L2 正規化再內積）
#   TF、DF、IDF、cosine 的數學沿用 Project 3 的 weighting/schemes.py
#
# 方法 B：TF-IDF Weighted Word Embedding
#   每個詞有一個 100 維的 Word2Vec 向量 v(t)
#   文件向量 = Σ w(t,d)·v(t) / Σ w(t,d)        （用 TF-IDF 權重做加權平均）
#   查詢向量也用同樣方式，分數 = cosine(q, d)
#   只有 Word2Vec 詞彙表裡的詞會被使用（訓練語料出現少於 3 次的詞沒有向量）
#
# 方法 C（額外）：Mean Embedding — 不加權的平均詞向量，用來看「TF-IDF 加權」本身有沒有幫助
# 方法 D（額外）：Hybrid — score = 0.5 × cosine_A + 0.5 × cosine_B
#
# 查詢詞的 idf：若查詢詞沒有出現在測試集（df = 0），TF-IDF 無法使用它；
#   方法 B 仍可用它的詞向量，這時把它當成最稀有的詞，idf = log10(N / 1)
# =============================================================================
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy import sparse

from weighting import schemes

from . import embedding
from .corpus import TOPICS, document_text, split_documents
from .text import preprocess

HYBRID_ALPHA = 0.5

# 方法代號 → 顯示名稱（順序就是網頁、CSV、圖表的顯示順序）
METHODS = {
    'tfidf': 'A · TF-IDF',
    'w2v_tfidf_cbow': 'B · TF-IDF weighted CBOW',
    'w2v_tfidf_sg': 'B · TF-IDF weighted Skip-Gram',
    'w2v_mean_cbow': 'C · Mean CBOW',
    'w2v_mean_sg': 'C · Mean Skip-Gram',
    'hybrid_cbow': 'D · Hybrid A + CBOW',
    'hybrid_sg': 'D · Hybrid A + Skip-Gram',
}
CORE_METHODS = ('tfidf', 'w2v_tfidf_cbow', 'w2v_tfidf_sg')


class ReutersIndex:
    def __init__(self, w2v_corpus='train'):
        self.w2v_corpus = w2v_corpus
        self.documents = split_documents('test')
        self.n = len(self.documents)
        self.doc_terms = [preprocess(document_text(doc)) for doc in self.documents]
        vocabulary = {}
        rows, cols, values = [], [], []
        for row, terms in enumerate(self.doc_terms):
            for term, tf in Counter(terms).items():
                rows.append(row)
                cols.append(vocabulary.setdefault(term, len(vocabulary)))
                values.append(tf)
        self.vocabulary = vocabulary
        self.terms = sorted(vocabulary, key=vocabulary.get)
        # tf 稀疏矩陣：3,299 篇 × 詞彙量
        self.counts = sparse.csr_matrix((values, (rows, cols)), shape=(self.n, len(vocabulary)), dtype=np.float64)
        self.df = schemes.document_frequency(self.counts)
        self.idf = schemes.inverse_document_frequency(self.df, self.n)       # log10(N / df)
        self.lengths = np.asarray(self.counts.sum(axis=1)).ravel()
        # 方法 A：TF-IDF 矩陣，每列 L2 正規化 → 內積 = cosine
        self.tfidf = schemes.l2_normalize_rows(schemes.weight_matrix(self.counts, 'tfidf', self.idf))
        self._embedding_matrices = {}
        # NEWID → 列編號（新聞內頁用）
        self.row_of = {doc['id']: row for row, doc in enumerate(self.documents)}

    # 某篇新聞在某方法的完整排名中排第幾（1 起算；沒被找到回傳 None）
    def rank_of(self, query, method, row):
        for rank, (other, _) in enumerate(self.ranking(query, method, limit=None), 1):
            if other == row:
                return rank
        return None

    # 某篇新聞 TF-IDF 權重最高的詞：[(詞, tf, idf, tf × idf), ...]
    def top_terms(self, row, n=10):
        counts = self.counts.getrow(row).tocoo()
        items = [(self.terms[col], int(tf), float(self.idf[col]), float(tf * self.idf[col]))
                 for col, tf in zip(counts.col, counts.data)]
        return sorted(items, key=lambda item: (-item[3], item[0]))[:n]

    # 一個詞的 idf；不在測試集的詞視為最稀有（df = 1）
    def term_idf(self, term):
        col = self.vocabulary.get(term)
        return float(self.idf[col]) if col is not None else float(np.log10(self.n))

    # 查詢 → TF-IDF 向量（與文件同一個詞彙空間；不在詞彙表的詞忽略）
    def query_tfidf(self, terms):
        vector = np.zeros(len(self.vocabulary))
        for term, tf in Counter(terms).items():
            col = self.vocabulary.get(term)
            if col is not None:
                vector[col] = tf * self.idf[col]
        return schemes.l2_normalize_vector(vector)

    # 把一串詞加權平均成一個詞向量；weighted=True 用 TF-IDF 權重，False 用平均
    @staticmethod
    def _average(term_weights, vectors):
        total = np.zeros(vectors.vector_size)
        weight_sum = 0.0
        for term, weight in term_weights.items():
            if term in vectors.key_to_index and weight > 0:
                total += weight * vectors[term]
                weight_sum += weight
        return total / weight_sum if weight_sum else total

    # 每個詞的權重：weighted=True 用 tf × idf（方法 B），False 每個詞都是 1（方法 C）
    def _term_weights(self, terms, weighted):
        counts = Counter(terms)
        if not weighted:
            return {term: 1.0 for term in counts}
        return {term: tf * self.term_idf(term) for term, tf in counts.items()}

    # 方法 B / C 的文件矩陣（每列已 L2 正規化），每種組合只算一次
    def embedding_matrix(self, architecture, weighted):
        key = (architecture, weighted)
        if key not in self._embedding_matrices:
            vectors, _ = embedding.load(self.w2v_corpus, architecture)
            matrix = np.vstack([self._average(self._term_weights(terms, weighted), vectors)
                                for terms in self.doc_terms])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._embedding_matrices[key] = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        return self._embedding_matrices[key]

    def query_embedding(self, terms, architecture, weighted):
        vectors, _ = embedding.load(self.w2v_corpus, architecture)
        return schemes.l2_normalize_vector(self._average(self._term_weights(terms, weighted), vectors))

    # 對全部 3,299 篇計分（cosine），回傳長度 N 的陣列
    def scores(self, query, method):
        terms = preprocess(query)
        if method == 'tfidf':
            return np.asarray(self.tfidf @ self.query_tfidf(terms)).ravel()
        family, _, architecture = method.rpartition('_')
        architecture = 'skipgram' if architecture == 'sg' else 'cbow'
        if family == 'hybrid':
            return HYBRID_ALPHA * self.scores(query, 'tfidf') + (1 - HYBRID_ALPHA) * self.scores(
                query, 'w2v_tfidf_' + method.rsplit('_', 1)[1])
        weighted = family == 'w2v_tfidf'
        return self.embedding_matrix(architecture, weighted) @ self.query_embedding(terms, architecture, weighted)

    # 依分數由高到低排序；空白文件（向量為 0）不列入
    def ranking(self, query, method, limit=10):
        scores = self.scores(query, method)
        candidates = np.flatnonzero(self.lengths > 0)
        order = candidates[np.lexsort((candidates, -scores[candidates]))]
        if method == 'tfidf':
            order = order[scores[order] > 0]       # TF-IDF：完全沒有共同詞的文件不算「找到」
        return [(int(row), float(scores[row])) for row in (order if limit is None else order[:limit])]

    # 查詢詞的診斷資訊：哪些詞在測試集裡（TF-IDF 用得到）、哪些詞有詞向量（Embedding 用得到）
    def query_terms(self, query, architecture='cbow'):
        vectors, _ = embedding.load(self.w2v_corpus, architecture)
        return [{'term': term, 'in_collection': term in self.vocabulary,
                 'df': int(self.df[self.vocabulary[term]]) if term in self.vocabulary else 0,
                 'idf': self.term_idf(term), 'has_vector': term in vectors.key_to_index}
                for term in dict.fromkeys(preprocess(query))]

    def summary(self):
        topic_counts = {topic: sum(topic in doc['topics'] for doc in self.documents) for topic in TOPICS}
        return {'documents': self.n, 'tokens': int(self.lengths.sum()), 'vocabulary': len(self.vocabulary),
                'avg_length': float(self.lengths.mean()), 'empty': int((self.lengths == 0).sum()),
                'topic_counts': topic_counts,
                'other_topics': sum(not any(t in TOPICS for t in doc['topics']) for doc in self.documents)}


@lru_cache(maxsize=2)
def load_index(w2v_corpus='train'):
    return ReutersIndex(w2v_corpus)
