# =============================================================================
# reuters/evaluation.py
# Project 4：檢索評估（以 Reuters topics 欄位判斷相關性）
#
#   相關性：新聞的 topics 含有查詢對應的主題 → 1，否則 0（二元）
#   指標（公式與 Project 3 相同，直接沿用 weighting/evaluation.py）：
#     Precision@10 = 前 10 名中相關文件數 / 10
#     MRR          = 1 / 第一篇相關文件的名次，再對所有查詢平均
#     nDCG@10      = DCG@10 / IDCG@10，DCG = Σ (2^rel − 1) / log2(rank + 1)；
#                    二元相關時每篇相關文件的 gain 都是 1，只剩「位置折扣」的效果
#     Recall@10 一併計算，但相關文件有 117–1,087 篇，上限很低，僅供參考
# =============================================================================
import numpy as np

from weighting.evaluation import ndcg_at, precision_at, recall_at, reciprocal_rank

from .queries import EVALUATION_QUERIES
from .retrieval import METHODS

METRICS = {'p10': 'P@10', 'mrr': 'MRR', 'ndcg10': 'nDCG@10', 'r10': 'Recall@10'}


# 每篇測試文件對某個主題的相關等級（0 / 1）
def relevance(index, topic):
    return np.array([1 if topic in doc['topics'] else 0 for doc in index.documents], dtype=np.int8)


def metrics_for(ranked_rows, grades):
    return {'p10': precision_at(ranked_rows, grades, 10), 'mrr': reciprocal_rank(ranked_rows, grades),
            'ndcg10': ndcg_at(ranked_rows, grades, 10), 'r10': recall_at(ranked_rows, grades, 10)}


# 完整評估：每個查詢 × 每個方法；回傳逐查詢結果、各方法平均、依查詢類型與主題的平均
def evaluate_all(index, methods=None, queries=EVALUATION_QUERIES):
    methods = list(methods or METHODS)
    per_query = []
    for query in queries:
        grades = relevance(index, query['topic'])
        for method in methods:
            # MRR 需要完整排名，所以 limit=None
            ranked = [row for row, _ in index.ranking(query['text'], method, limit=None)]
            per_query.append({'query_id': query['id'], 'topic': query['topic'], 'type': query['type'],
                              'method': method, 'label': METHODS[method], **metrics_for(ranked, grades)})

    def average(rows):
        return {key: float(np.mean([row[key] for row in rows])) if rows else 0.0 for key in METRICS}

    means = [{'method': m, 'label': METHODS[m], **average([r for r in per_query if r['method'] == m])} for m in methods]
    by_type = [{'method': m, 'type': t, **average([r for r in per_query if r['method'] == m and r['type'] == t])}
               for m in methods for t in ('keyword', 'paraphrase')]
    by_topic = [{'method': m, 'topic': t, **average([r for r in per_query if r['method'] == m and r['topic'] == t])}
                for m in methods for t in dict.fromkeys(q['topic'] for q in queries)]
    return {'per_query': per_query, 'means': means, 'by_type': by_type, 'by_topic': by_topic}
