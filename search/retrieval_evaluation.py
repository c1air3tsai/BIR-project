"""Judged-only evaluation over a union of four Top-10 result pools."""
import math
from statistics import mean

from .models import EvaluationQuery, RelevanceJudgment
from .term_weighting import METHODS

METRIC_NAMES = ('p5', 'p10', 'r10', 'mrr', 'ndcg10')


def precision_at(ranked_ids, qrels, k):
    return sum(qrels.get(doc_id, 0) > 0 for doc_id in ranked_ids[:k]) / k


def recall_at(ranked_ids, qrels, k):
    relevant = sum(grade > 0 for grade in qrels.values())
    return sum(qrels.get(doc_id, 0) > 0 for doc_id in ranked_ids[:k]) / relevant if relevant else 0.0


def reciprocal_rank(ranked_ids, qrels):
    return next((1 / rank for rank, doc_id in enumerate(ranked_ids, 1)
                 if qrels.get(doc_id, 0) > 0), 0.0)


def dcg_at(ranked_ids, qrels, k):
    return sum((2 ** qrels.get(doc_id, 0) - 1) / math.log2(rank + 1)
               for rank, doc_id in enumerate(ranked_ids[:k], 1))


def ndcg_at(ranked_ids, qrels, k):
    ideal = sorted(qrels, key=lambda doc_id: -qrels[doc_id])
    denominator = dcg_at(ideal, qrels, k)
    return dcg_at(ranked_ids, qrels, k) / denominator if denominator else 0.0


def metrics(ranked_ids, qrels):
    return {'p5': precision_at(ranked_ids, qrels, 5),
            'p10': precision_at(ranked_ids, qrels, 10),
            'r10': recall_at(ranked_ids, qrels, 10),
            'mrr': reciprocal_rank(ranked_ids, qrels),
            'ndcg10': ndcg_at(ranked_ids, qrels, 10)}


def candidate_pool(index, query):
    comparison = index.compare(query)
    ids = [row['id'] for row in comparison['rows']]
    documents = {doc['id']: doc for doc in index.documents}
    return [{'id': doc_id, 'title': documents[doc_id]['title'],
             'snippet': documents[doc_id]['abstract'][:240],
             'matched': [term for term in index.query_weights(query, 'tf')[0]
                         if term in documents[doc_id]['terms']]}
            for doc_id in ids], comparison


def evaluate(index, include_detail=False):
    queries = list(EvaluationQuery.objects.order_by('id').values('id', 'text'))
    labels = {}
    for query_id, document_id, relevance in RelevanceJudgment.objects.filter(
            query_id__in=[row['id'] for row in queries]).values_list('query_id', 'document_id', 'relevance'):
        labels.setdefault(query_id, {})[document_id] = relevance
    rows, ready = [], []
    for query in queries:
        pool, comparison = candidate_pool(index, query['text'])
        pool_ids = {row['id'] for row in pool}
        qrels = {doc_id: grade for doc_id, grade in labels.get(query['id'], {}).items()
                 if doc_id in pool_ids}
        complete = bool(pool_ids) and pool_ids <= qrels.keys()
        row = {'id': query['id'], 'text': query['text'], 'judged': len(qrels),
               'pool': len(pool_ids), 'relevant': sum(value > 0 for value in qrels.values()),
               'ready': complete}
        if complete:
            row['metrics'] = {method: metrics([item['id'] for item in comparison['rankings'][method]], qrels)
                              for method in METHODS}
            ready.append(row)
        if include_detail:
            row['rankings'] = {method: [{**item, 'relevance': qrels.get(item['id'])}
                                        for item in comparison['rankings'][method]] for method in METHODS}
            row['pool_documents'] = [{**item, 'relevance': qrels.get(item['id'])} for item in pool]
        rows.append(row)
    aggregate = {method: {name: mean(row['metrics'][method][name] for row in ready) if ready else None
                          for name in METRIC_NAMES} for method in METHODS}
    winners = {name: [method for method in METHODS
                      if ready and abs(aggregate[method][name] - max(aggregate[m][name] for m in METHODS)) < 1e-12]
               for name in METRIC_NAMES}
    return {'queries': rows, 'ready': len(ready), 'total': len(rows),
            'methods': list(METHODS), 'aggregate': aggregate, 'winners': winners,
            'judgment_source': 'manual'}
