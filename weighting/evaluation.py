# =============================================================================
# weighting/evaluation.py
# Project 3 任務五：搜尋結果評估
#
#   1. relevance_grades()：依 weighting/queries.py 的 MeSH 規則，替每篇文件打 0 / 1 / 2 分
#   2. 指標（ranked = 系統排名的文件列編號，grades = 每篇文件的相關等級）：
#        Precision@k = 前 k 名中相關文件數 / k
#        Recall@k    = 前 k 名中相關文件數 / 語料中所有相關文件數
#        MRR         = 1 / 第一篇相關文件的名次（沒找到為 0），再對所有查詢取平均
#        nDCG@k      = DCG@k / IDCG@k
#                      DCG@k = Σ (2^rel − 1) / log2(rank + 1)，IDCG 為理想排序的 DCG
#   3. evaluate_all()：每個查詢 × 每種方法 × 有無 L2 都算一次，並計算平均
# =============================================================================
import math

import numpy as np

from .queries import EVALUATION_QUERIES
from .schemes import METHODS

METRICS = {'p5': 'P@5', 'p10': 'P@10', 'r10': 'Recall@10', 'mrr': 'MRR', 'ndcg10': 'nDCG@10'}


# 依 MeSH 規則替每篇文件打分；回傳長度 N 的整數陣列（0 / 1 / 2）
def relevance_grades(index, query):
    descriptors = set(query.get('mesh', []))
    qualifiers = set(query.get('qualifiers', []))
    grades = np.zeros(index.n_documents, dtype=np.int8)
    for row, headings in enumerate(index.mesh):
        for heading in headings:
            hit = heading['descriptor'] in descriptors or qualifiers.intersection(heading['qualifiers'])
            if hit:
                grades[row] = max(grades[row], 2 if heading['major'] else 1)
    return grades


def precision_at(ranked, grades, k):
    return sum(grades[row] > 0 for row in ranked[:k]) / k


def recall_at(ranked, grades, k):
    total_relevant = int((grades > 0).sum())
    return sum(grades[row] > 0 for row in ranked[:k]) / total_relevant if total_relevant else 0.0


# Reciprocal Rank：第一篇相關文件在第 r 名 → 1/r
def reciprocal_rank(ranked, grades):
    for rank, row in enumerate(ranked, 1):
        if grades[row] > 0:
            return 1.0 / rank
    return 0.0


# DCG@k = Σ_{rank=1..k} (2^rel − 1) / log2(rank + 1)
def dcg(relevances, k):
    return sum((2 ** int(rel) - 1) / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], 1))


# nDCG@k = DCG(系統排序) / DCG(把所有文件依相關等級由高到低排的理想排序)
def ndcg_at(ranked, grades, k):
    ideal = dcg(sorted(grades.tolist(), reverse=True), k)
    return dcg([grades[row] for row in ranked], k) / ideal if ideal else 0.0


# 一個查詢、一種排名的全部指標
def metrics_for(ranked, grades):
    return {'p5': precision_at(ranked, grades, 5), 'p10': precision_at(ranked, grades, 10),
            'r10': recall_at(ranked, grades, 10), 'mrr': reciprocal_rank(ranked, grades),
            'ndcg10': ndcg_at(ranked, grades, 10)}


# 所有實驗組合：(方法, 是否 L2)
def configurations(methods=None):
    return [(method, use_l2) for method in (methods or METHODS) for use_l2 in (False, True)]


def config_label(method, use_l2):
    return f"{METHODS[method]} {'+ L2' if use_l2 else '(no L2)'}"


# 執行完整評估
# 回傳：
#   queries   每個查詢的相關文件數（2 分 / 1 分）
#   per_query 每個查詢 × 每個組合的指標
#   means     每個組合在所有查詢上的平均指標
def evaluate_all(index, queries=EVALUATION_QUERIES):
    query_rows, per_query = [], []
    for query in queries:
        grades = relevance_grades(index, query)
        query_rows.append({'id': query['id'], 'text': query['text'],
                           'rule': ', '.join(query.get('mesh', []) + [f'[qualifier] {q}' for q in query.get('qualifiers', [])]),
                           'relevant': int((grades > 0).sum()), 'highly_relevant': int((grades == 2).sum())})
        for method, use_l2 in configurations():
            # MRR 需要完整排名（第一篇相關文件可能在第 10 名之後），所以 limit=None
            ranked = index.ranking(query['text'], method, use_l2, limit=None)
            per_query.append({'query_id': query['id'], 'method': method, 'use_l2': use_l2,
                              'label': config_label(method, use_l2), **metrics_for(ranked, grades)})
    means = []
    for method, use_l2 in configurations():
        rows = [row for row in per_query if row['method'] == method and row['use_l2'] == use_l2]
        means.append({'method': method, 'use_l2': use_l2, 'label': config_label(method, use_l2),
                      **{name: float(np.mean([row[name] for row in rows])) if rows else 0.0 for name in METRICS}})
    return {'queries': query_rows, 'per_query': per_query, 'means': means}
