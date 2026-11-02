# =============================================================================
# weighting/index.py
# Project 3 任務一＋任務三：建立向量空間索引、詞項統計、搜尋排名
#
# 資料來源：資料庫中所有「有 PMID、有摘要」的 PubMed 文件（GLP-1 主題，目前 1000 篇），
#           只使用 abstract（含 Keywords 行），與前幾次作業的檢索範圍一致。
#
# 快取設計（避免每次搜尋都重算）：
#   - 第一次使用時把每篇摘要前處理 → 建立 tf 稀疏矩陣（文件 × 詞彙）
#   - 存成 data/retrieval_cache/weighting/<前處理代號>.npz（矩陣）＋ .json（詞彙、文件 ID、MeSH）
#   - 之後直接讀檔；資料庫文件數或更新時間改變時，簽章不同 → 自動重建
#   - 讀進記憶體後，每種「方法 × 有無 L2」的權重矩陣只算一次，存在 self._matrices
#
# 主要對外函式：
#   load_index(config)       取得（必要時建立）某個前處理設定的索引
#   WeightingIndex.search()  查詢 → 向量 → 分數 → Top-k
#   WeightingIndex.term_info()、top_df_terms()、length_histogram()  詞項統計
# =============================================================================
import json
import os
import re
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from django.conf import settings
from django.db.models import Count, Max
from scipy import sparse

from search.models import Document

from . import schemes
from .preprocessing import DEFAULT_CONFIG, PreprocessConfig, preprocess

# 演算法或資料格式改變時改這個版本號，舊快取就會自動失效
INDEX_VERSION = 'p3-weighting-v2'

# HW1 的 XML 解析會在摘要最後加一行 "Keywords: a, b, c"；
# 關鍵詞本身保留，但 "Keywords" 這個標籤字不是內容，建索引前先拿掉
_KEYWORDS_LABEL = re.compile(r'(?<![\w-])Keywords?\s*:', re.IGNORECASE)


def index_text(abstract):
    return _KEYWORDS_LABEL.sub(' ', abstract or '')


# 本專案使用的文件集合：有 PMID、有摘要的 PubMed 文件（排除 arXiv CS 文件）
def corpus_queryset():
    return (Document.objects.exclude(abstract='').exclude(pmid='')
            .filter(arxiv_id='').order_by('pk'))


# 用「文件數 + 最後更新時間」當作資料簽章，用來判斷快取是否過期
def corpus_signature():
    stats = corpus_queryset().aggregate(count=Count('pk'), latest=Max('indexed_at'))
    latest = stats['latest'].isoformat() if stats['latest'] else ''
    return f"{INDEX_VERSION}|{stats['count']}|{latest}"


def cache_dir():
    base = getattr(settings, 'RETRIEVAL_CACHE_DIR', Path(settings.BASE_DIR) / 'data' / 'retrieval_cache')
    return Path(base) / 'weighting'


# ---------------------------------------------------------------------------
# MeSH 解析（任務五的相關性判斷要用）
# ---------------------------------------------------------------------------

# 從 PubMed XML 讀出 MeSH headings
# 回傳 [{'descriptor': 'Obesity', 'major': True, 'qualifiers': ['drug therapy']}, ...]
# major = 描述詞本身或任何一個 qualifier 被標成 MajorTopicYN="Y"
def parse_mesh(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return []
    headings = []
    for heading in root.iter('MeshHeading'):
        descriptor = heading.find('DescriptorName')
        if descriptor is None or not descriptor.text:
            continue
        qualifiers = heading.findall('QualifierName')
        major = descriptor.get('MajorTopicYN') == 'Y' or any(q.get('MajorTopicYN') == 'Y' for q in qualifiers)
        headings.append({'descriptor': descriptor.text.strip(), 'major': major,
                         'qualifiers': [q.text.strip() for q in qualifiers if q.text]})
    return headings


# ---------------------------------------------------------------------------
# 建立 / 讀取快取
# ---------------------------------------------------------------------------

def _paths(config):
    base = cache_dir()
    return base / f'{config.key}.npz', base / f'{config.key}.json'


# 先寫到暫存檔再改名，避免寫到一半的檔案被讀到
def _atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(handle)
    try:
        write(temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


# 從資料庫重新建立 tf 矩陣並存檔
def build_index(config=DEFAULT_CONFIG):
    corpus_dir = Path(settings.BASE_DIR) / 'data' / 'corpus'
    documents = list(corpus_queryset().values_list('pk', 'pmid', 'abstract', 'source_file'))
    vocabulary = {}
    rows, cols, values, doc_ids, pmids, mesh = [], [], [], [], [], []
    for pk, pmid, abstract, source_file in documents:
        counts = Counter(preprocess(index_text(abstract), config))
        if not counts:
            continue
        row = len(doc_ids)
        doc_ids.append(pk)
        pmids.append(pmid)
        mesh.append(parse_mesh(corpus_dir / source_file))
        for term, tf in counts.items():
            # setdefault：第一次看到的詞給它一個新的欄位編號
            rows.append(row)
            cols.append(vocabulary.setdefault(term, len(vocabulary)))
            values.append(tf)
    counts_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(doc_ids), len(vocabulary)), dtype=np.int32)
    meta = {'signature': corpus_signature(), 'config': config.key,
            'vocabulary': sorted(vocabulary, key=vocabulary.get),
            'doc_ids': doc_ids, 'pmids': pmids, 'mesh': mesh}
    matrix_path, meta_path = _paths(config)
    _atomic_write(matrix_path, lambda name: sparse.save_npz(name, counts_matrix))
    _atomic_write(meta_path, lambda name: Path(name).write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8'))
    _load_cached.cache_clear()
    return meta


# 快取是否存在且簽章與目前資料庫一致
def index_ready(config=DEFAULT_CONFIG):
    matrix_path, meta_path = _paths(config)
    if not (matrix_path.is_file() and meta_path.is_file()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return meta.get('signature') == corpus_signature()


# 讀檔後放在記憶體（lru_cache），同一個簽章只讀一次
@lru_cache(maxsize=4)
def _load_cached(config, signature):
    matrix_path, meta_path = _paths(config)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    return WeightingIndex(config, sparse.load_npz(matrix_path).tocsr(), meta)


# 對外入口：取得索引；快取不存在或過期就自動重建
def load_index(config=DEFAULT_CONFIG, force_rebuild=False):
    if force_rebuild or not index_ready(config):
        build_index(config)
    return _load_cached(config, corpus_signature())


# 解析網址參數 ?stop=1&stem=0 → PreprocessConfig
def config_from_flags(stop, stem):
    return PreprocessConfig(remove_stopwords=str(stop) != '0', stemming=str(stem) == '1')


# ---------------------------------------------------------------------------
# 索引本體
# ---------------------------------------------------------------------------

class WeightingIndex:
    def __init__(self, config, counts, meta):
        self.config = config
        self.counts = counts                                   # N × V 的 tf 矩陣
        self.vocabulary = meta['vocabulary']                   # 欄位編號 → 詞
        self.term_to_col = {term: col for col, term in enumerate(self.vocabulary)}
        self.doc_ids = meta['doc_ids']                         # 列編號 → Document.pk
        self.pmids = meta['pmids']
        self.mesh = meta['mesh']
        self.n_documents = counts.shape[0]                     # N
        self.df = schemes.document_frequency(counts)           # df(t)
        self.cf = schemes.collection_frequency(counts)         # cf(t)
        self.idf = schemes.inverse_document_frequency(self.df, self.n_documents)  # log10(N/df)
        self.bm25_idf = schemes.bm25_idf(self.df, self.n_documents)
        self.doc_lengths = np.asarray(counts.sum(axis=1)).ravel()   # |d| = 前處理後 token 數
        self.avg_length = float(self.doc_lengths.mean()) if self.n_documents else 0.0
        self._matrices = {}
        self._documents = None

    # 取得某方法的文件權重矩陣；use_l2=True 時回傳每列已正規化的版本（只算一次就快取）
    def matrix(self, method, use_l2=False):
        key = (method, use_l2)
        if key not in self._matrices:
            if use_l2:
                self._matrices[key] = schemes.l2_normalize_rows(self.matrix(method, False))
            else:
                self._matrices[key] = schemes.weight_matrix(
                    self.counts, method, self.idf, self.doc_lengths, self.avg_length, self.bm25_idf)
        return self._matrices[key]

    # 文件的標題、摘要等顯示資料（第一次需要時才從資料庫讀）
    def documents(self):
        if self._documents is None:
            by_id = {row['pk']: row for row in Document.objects.filter(pk__in=self.doc_ids).values(
                'pk', 'pmid', 'title', 'abstract', 'publication_year', 'journal')}
            self._documents = [by_id.get(pk, {'pk': pk, 'pmid': '', 'title': '', 'abstract': '',
                                              'publication_year': '', 'journal': ''})
                               for pk in self.doc_ids]
        return self._documents

    # 查詢 → 與文件相同的前處理 → 詞彙空間中的 tf 向量
    # 回傳 (query_counts 向量, 查詢詞清單, 不在詞彙表中的詞)
    def query_counts(self, query):
        terms = preprocess(query, self.config)
        vector = np.zeros(len(self.vocabulary))
        unknown = []
        for term, tf in Counter(terms).items():
            col = self.term_to_col.get(term)
            if col is None:
                unknown.append(term)
            else:
                vector[col] = tf
        known = [term for term in dict.fromkeys(terms) if term in self.term_to_col]
        return vector, known, unknown

    # 對所有文件計分；回傳長度 N 的分數陣列
    def score_all(self, query_vector_counts, method, use_l2):
        query_vector = schemes.query_weights(query_vector_counts, method, self.idf, self.bm25_idf)
        return schemes.score_documents(self.matrix(method, use_l2), query_vector, use_l2)

    # 任務三主流程：查詢向量化 → 計算分數 → 排序 → Top-k
    def search(self, query, method='logtfidf', use_l2=True, limit=10):
        counts, known, unknown = self.query_counts(query)
        scores = self.score_all(counts, method, use_l2)
        ranked = schemes.rank_scores(scores, limit)
        return {'query': query, 'method': method, 'use_l2': use_l2, 'terms': known, 'unknown': unknown,
                'matches': int((scores > 0).sum()), 'ranked': ranked}

    # 只回傳排名後的文件列編號（評估用，速度快）
    def ranking(self, query, method, use_l2, limit=10):
        counts, _, _ = self.query_counts(query)
        return [row for row, _ in schemes.rank_scores(self.score_all(counts, method, use_l2), limit)]

    # ---------------- 任務一：詞項統計 ----------------

    # 整體統計：文件數、token 總數、詞彙量、文件長度
    def summary(self):
        lengths = self.doc_lengths
        return {'documents': self.n_documents, 'tokens': int(lengths.sum()),
                'vocabulary': len(self.vocabulary),
                'avg_length': self.avg_length,
                'median_length': float(np.median(lengths)) if len(lengths) else 0.0,
                'min_length': int(lengths.min()) if len(lengths) else 0,
                'max_length': int(lengths.max()) if len(lengths) else 0,
                'hapax': int((self.cf == 1).sum())}

    # 一個詞的 CF / DF / IDF，以及 TF 最高的幾篇文件
    def term_info(self, raw_term, top_documents=5):
        terms = preprocess(raw_term, self.config)
        if not terms:
            return {'input': raw_term, 'term': None}
        term = terms[0]
        col = self.term_to_col.get(term)
        if col is None:
            return {'input': raw_term, 'term': term, 'found': False}
        column = self.counts.getcol(col).tocoo()
        tf_rows = sorted(zip(column.data.tolist(), column.row.tolist()), key=lambda item: (-item[0], item[1]))
        return {'input': raw_term, 'term': term, 'found': True,
                'cf': int(self.cf[col]), 'df': int(self.df[col]), 'idf': float(self.idf[col]),
                'documents': [{'row': row, 'pmid': self.pmids[row], 'tf': tf,
                               'length': int(self.doc_lengths[row]),
                               'log_tf': 1 + np.log10(tf), 'tfidf': tf * float(self.idf[col])}
                              for tf, row in tf_rows[:top_documents]]}

    # DF 最高（或最低）的 n 個詞
    def top_df_terms(self, n=20, lowest=False, min_df=1):
        cols = np.flatnonzero(self.df >= min_df)
        key = (self.df[cols], -self.cf[cols]) if lowest else (-self.df[cols], -self.cf[cols])
        order = cols[np.lexsort(key[::-1])][:n]
        return [self.term_row(col) for col in order]

    # 一個詞的統計列（表格用）
    def term_row(self, col):
        return {'term': self.vocabulary[col], 'cf': int(self.cf[col]), 'df': int(self.df[col]),
                'idf': float(self.idf[col]), 'df_ratio': self.df[col] / self.n_documents}

    # 文件長度分布（直方圖）：回傳每個區間的 [起, 迄) 與篇數
    def length_histogram(self, bins=12):
        if not self.n_documents:
            return []
        counts, edges = np.histogram(self.doc_lengths, bins=bins)
        return [{'start': int(edges[i]), 'end': int(edges[i + 1]), 'count': int(counts[i])}
                for i in range(len(counts))]
