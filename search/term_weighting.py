"""Transparent sparse vector-space retrieval for abstract-only experiments.

The persistent artifact contains document term counts.  Postings and four sets
of norms are rebuilt from that compact artifact on load, never from abstracts
on a normal search request.  All four methods share one vocabulary and IDF.
"""
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.db.models import Count, Max

from .analysis import condition_tokens
from .models import Document

METHODS = ('tf', 'log_tf', 'idf', 'tfidf')
INDEX_VERSION = 'condition-c-connectors-v1-weighting-v2'
CONNECTORS = str.maketrans({'‐': '-', '‑': '-', '–': '-'})


def tokens(text):
    """Condition C for documents and queries, with equivalent hyphens folded."""
    return [token.translate(CONNECTORS) for token in condition_tokens(text)['C']]


def raw_tf(count):
    return float(count)


def log_tf(count):
    return 1.0 + math.log(count) if count > 0 else 0.0


def idf_weight(n, df):
    return math.log(n / df) if n > 0 and df > 0 else 0.0


def weight(count, idf, method):
    if method == 'tf':
        return raw_tf(count)
    if method == 'log_tf':
        return log_tf(count)
    if method == 'idf':
        return idf if count > 0 else 0.0
    if method == 'tfidf':
        return raw_tf(count) * idf
    raise ValueError('Unknown weighting method')


def scope_queryset(scope):
    qs = Document.objects.exclude(abstract='')
    if scope == 'medical':
        return qs.filter(arxiv_id='')
    if scope == 'cs':
        return qs.exclude(arxiv_id='')
    if scope == 'all':
        return qs
    raise ValueError('Unknown corpus')


def cache_root():
    return Path(getattr(settings, 'RETRIEVAL_CACHE_DIR', settings.BASE_DIR / 'data' / 'retrieval_cache'))


def signature(scope):
    stats = scope_queryset(scope).aggregate(count=Count('pk'), latest=Max('indexed_at'))
    return {'count': stats['count'], 'latest': stats['latest'].isoformat() if stats['latest'] else None}


def _paths(scope):
    base = cache_root() / scope
    return base, base / 'current.json'


def _read_manifest(scope):
    _, pointer = _paths(scope)
    try:
        return json.loads(pointer.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def index_status(scope):
    current = signature(scope)
    manifest = _read_manifest(scope)
    ready = bool(manifest and manifest.get('signature') == current and
                 manifest.get('version') == INDEX_VERSION and
                 manifest.get('key') and
                 (cache_root() / scope / (manifest['key'] + '.json.gz')).is_file())
    return {'available': current['count'], 'indexed': manifest['documents'] if ready else 0,
            'vocabulary': manifest['vocabulary'] if ready else 0,
            'ready': ready, 'key': manifest['key'] if ready else None}


def _atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp_name = stream.name
        stream.write(content)
    os.replace(temp_name, path)


def build_index(scope='medical', force=False):
    started = time.perf_counter()
    before = index_status(scope)
    if before['ready'] and not force:
        return {**before, 'cache_hit': True, 'seconds': 0.0}
    rows = list(scope_queryset(scope).order_by('pk').values_list(
        'pk', 'title', 'abstract', 'publication_year', 'journal', 'arxiv_id'))
    digest = hashlib.sha256(f'{scope}\0{INDEX_VERSION}\0'.encode())
    documents, cf, df = [], Counter(), Counter()
    for pk, title, abstract, year, journal, arxiv_id in rows:
        digest.update(f'{pk}\0{title}\0{abstract}\0{year}\0{journal}\0{arxiv_id}\0'.encode('utf-8'))
        counts = Counter(tokens(abstract))
        if not counts:
            continue
        documents.append({'id': pk, 'title': title, 'abstract': abstract,
                          'year': year, 'source': journal or ('arXiv' if arxiv_id else ''),
                          'terms': dict(counts)})
        cf.update(counts)
        df.update(counts.keys())
    key = digest.hexdigest()
    artifact = {'version': INDEX_VERSION, 'scope': scope, 'key': key,
                'documents': documents, 'cf': dict(cf), 'df': dict(df)}
    base, pointer = _paths(scope)
    _atomic_bytes(base / (key + '.json.gz'), gzip.compress(
        json.dumps(artifact, ensure_ascii=False, separators=(',', ':')).encode('utf-8')))
    manifest = {'version': INDEX_VERSION, 'key': key, 'signature': signature(scope),
                'documents': len(documents), 'vocabulary': len(df)}
    _atomic_bytes(pointer, json.dumps(manifest, sort_keys=True).encode('utf-8'))
    _load_artifact.cache_clear()
    return {'available': len(rows), 'indexed': len(documents), 'vocabulary': len(df),
            'ready': True, 'key': key, 'cache_hit': False,
            'seconds': time.perf_counter() - started}


@lru_cache(maxsize=6)
def _load_artifact(scope, key):
    with gzip.open(cache_root() / scope / (key + '.json.gz'), 'rt', encoding='utf-8') as stream:
        return RetrievalIndex(json.load(stream))


def load_index(scope='medical', build=False):
    status = index_status(scope)
    if not status['ready'] and build:
        status = build_index(scope)
    return _load_artifact(scope, status['key']) if status['ready'] else None


class RetrievalIndex:
    def __init__(self, artifact):
        self.scope = artifact['scope']
        self.key = artifact['key']
        self.documents = artifact['documents']
        self.cf = artifact['cf']
        self.df = artifact['df']
        self.n = len(self.documents)
        self.idf = {term: idf_weight(self.n, frequency) for term, frequency in self.df.items()}
        self.postings = defaultdict(list)
        self.norms = {method: [] for method in METHODS}
        for row, document in enumerate(self.documents):
            sums = {method: 0.0 for method in METHODS}
            for term, count in document['terms'].items():
                self.postings[term].append((row, count))
                for method in METHODS:
                    value = weight(count, self.idf[term], method)
                    sums[method] += value * value
            for method in METHODS:
                self.norms[method].append(math.sqrt(sums[method]))

    def query_weights(self, query, method):
        return self._weights_for_counts(Counter(tokens(query)), method)

    def _weights_for_counts(self, counts, method):
        oov = sorted(set(counts) - self.df.keys())
        values = {term: weight(count, self.idf[term], method)
                  for term, count in counts.items() if term in self.df}
        return {term: value for term, value in values.items() if value}, oov

    def rank(self, query, method='tfidf', limit=10):
        if method not in METHODS:
            raise ValueError('Unknown weighting method')
        start = time.perf_counter()
        counts = Counter(tokens(query))
        preprocessed = time.perf_counter()
        qweights, oov = self._weights_for_counts(counts, method)
        vectorized = time.perf_counter()
        qnorm = math.sqrt(sum(value * value for value in qweights.values()))
        numerator = defaultdict(float)
        if qnorm:
            for term, qweight in qweights.items():
                for row, count in self.postings[term]:
                    numerator[row] += qweight * weight(count, self.idf[term], method)
        similarities = time.perf_counter()
        scored = [(score / (qnorm * self.norms[method][row]), row)
                  for row, score in numerator.items() if score > 0 and self.norms[method][row] > 0]
        scored.sort(key=lambda item: (-item[0], self.documents[item[1]]['id']))
        ranked_at = time.perf_counter()
        ranked = []
        for rank, (score, row) in enumerate(scored[:limit], 1):
            doc = self.documents[row]
            ranked.append({'rank': rank, 'id': doc['id'], 'title': doc['title'],
                           'score': score, 'snippet': doc['abstract'][:240],
                           'year': doc['year'], 'source': doc['source'],
                           'matched': [term for term in qweights if term in doc['terms']]})
        return {'query': query, 'method': method, 'results': ranked, 'oov': oov,
                'tokens': list(qweights), 'total_matches': len(scored),
                'timing': {'preprocess': preprocessed - start,
                           'vectorize': vectorized - preprocessed,
                           'similarity': similarities - vectorized,
                           'ranking': ranked_at - similarities}}

    def explain(self, query, method, document_id):
        qweights, _ = self.query_weights(query, method)
        qnorm = math.sqrt(sum(value * value for value in qweights.values()))
        row = next((i for i, doc in enumerate(self.documents) if doc['id'] == document_id), None)
        if row is None:
            raise ValueError('Document is outside the selected index')
        document = self.documents[row]
        details = []
        for term, qweight in qweights.items():
            count = document['terms'].get(term, 0)
            if not count:
                continue
            dweight = weight(count, self.idf[term], method)
            details.append({'term': term, 'query_weight': qweight, 'tf': count,
                            'cf': self.cf[term], 'df': self.df[term], 'idf': self.idf[term],
                            'document_weight': dweight, 'contribution': qweight * dweight})
        details.sort(key=lambda item: (-item['contribution'], item['term']))
        denominator = qnorm * self.norms[method][row]
        return {'document_id': document_id, 'method': method, 'terms': details,
                'query_norm': qnorm, 'document_norm': self.norms[method][row],
                'numerator': sum(item['contribution'] for item in details),
                'score': sum(item['contribution'] for item in details) / denominator if denominator else 0.0}

    def inspect_terms(self, text):
        return [{'term': term, 'cf': self.cf.get(term, 0), 'df': self.df.get(term, 0),
                 'idf': self.idf.get(term)} for term in dict.fromkeys(tokens(text))]

    def compare(self, query):
        rankings = {method: self.rank(query, method)['results'] for method in METHODS}
        ids = {method: [row['id'] for row in results] for method, results in rankings.items()}
        union = list(dict.fromkeys(doc_id for method in METHODS for doc_id in ids[method]))
        titles = {doc['id']: doc['title'] for doc in self.documents}
        rows = [{'id': doc_id, 'title': titles[doc_id],
                 'ranks': {method: ids[method].index(doc_id) + 1 if doc_id in ids[method] else None
                           for method in METHODS}} for doc_id in union]
        raw_first = next((doc for doc in self.documents
                          if ids['tf'] and doc['id'] == ids['tf'][0]), None)
        term_evidence = {'document_id': raw_first['id'], 'title': raw_first['title'],
                         'terms': [{'term': term, 'tf': raw_first['terms'][term],
                                    'raw_weight': raw_tf(raw_first['terms'][term]),
                                    'log_weight': log_tf(raw_first['terms'][term]),
                                    'df': self.df[term], 'idf': self.idf[term]}
                                   for term in self.query_weights(query, 'tf')[0]
                                   if term in raw_first['terms']]} if raw_first else None
        a, b = set(ids['tf']), set(ids['log_tf'])
        return {'query': query, 'methods': list(METHODS), 'rows': rows,
                'rankings': rankings, 'term_evidence': term_evidence, 'summary': {
                    'overlap': len(a & b), 'jaccard': len(a & b) / len(a | b) if a | b else 1.0,
                    'shared_first': bool(ids['tf'] and ids['log_tf'] and ids['tf'][0] == ids['log_tf'][0]),
                    'largest_shared_rank_change': max((abs(ids['tf'].index(i) - ids['log_tf'].index(i))
                                                       for i in a & b), default=0)}}

    def l2_experiment(self, query, method='tfidf'):
        ordinary = self.rank(query, method)['results']
        qweights, _ = self.query_weights(query, method)
        qnorm = math.sqrt(sum(value * value for value in qweights.values()))
        explicit = []
        if qnorm:
            for row, doc in enumerate(self.documents):
                dnorm = self.norms[method][row]
                if not dnorm:
                    continue
                score = sum((qweight / qnorm) * (weight(doc['terms'].get(term, 0), self.idf[term], method) / dnorm)
                            for term, qweight in qweights.items())
                if score > 0:
                    explicit.append((score, doc['id']))
        explicit.sort(key=lambda item: (-item[0], item[1]))
        top = explicit[:10]
        by_id = {doc_id: score for score, doc_id in top}
        return {'overlap': len(set(row['id'] for row in ordinary) & set(by_id)),
                'same_order': [row['id'] for row in ordinary] == [doc_id for _, doc_id in top],
                'max_score_difference': max((abs(row['score'] - by_id[row['id']])
                                             for row in ordinary if row['id'] in by_id), default=0.0),
                'raw_ids': [row['id'] for row in ordinary], 'normalized_ids': [doc_id for _, doc_id in top]}
