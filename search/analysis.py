"""Abstract-only, deterministic A–D comparisons and unsmoothed natural-log IDF."""
import hashlib
import math
import re
from collections import Counter, defaultdict

from django.core.cache import cache

from .text_processing import STOPWORDS, iter_token_matches, stem_tokens

CONDITIONS = {
    'A': ('Basic', 'Words and standalone punctuation · casefold'),
    'B': ('No punctuation', 'A with standalone punctuation removed'),
    'C': ('No stopwords', 'B − the shared HW1 stopword list'),
    'D': ('Porter stemming', 'C + Porter stemming'),
}


def iter_basic_spans(text):
    """Yield biomedical word tokens plus each standalone punctuation mark.

    Word matching runs first so punctuation that belongs inside a biomedical
    token (for example ``GLP-1``, ``0.05`` or ``patient's``) stays attached.
    Non-whitespace characters in the gaps are Condition A punctuation tokens.
    """
    source = text or ''
    cursor = 0
    for match in iter_token_matches(source):
        for index, char in enumerate(source[cursor:match.start()], cursor):
            if not char.isspace():
                yield index, index + 1, char.casefold()
        yield match.start(), match.end(), source[match.start():match.end()].casefold()
        cursor = match.end()
    for index, char in enumerate(source[cursor:], cursor):
        if not char.isspace():
            yield index, index + 1, char.casefold()


def basic_tokens(text):
    return [token for _, _, token in iter_basic_spans(text)]


def condition_tokens(text):
    a = basic_tokens(text or '')
    b = [token for token in a if any(char.isalnum() for char in token)]
    c = [token for token in b if token not in STOPWORDS]
    return {'A': a, 'B': b, 'C': c, 'D': stem_tokens(c)}


def fit_log_log(rows):
    if len(rows) < 2:
        return {'slope': None, 'intercept': None, 'exponent': None, 'r2': None, 'rmse': None}
    xs = [math.log(row['rank']) for row in rows]
    ys = [math.log(row['cf']) for row in rows]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    sxx = sum((x-mx)**2 for x in xs)
    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sxx
    intercept = my-slope*mx
    sse = sum((y-(intercept+slope*x))**2 for x,y in zip(xs,ys))
    syy = sum((y-my)**2 for y in ys)
    return {
        'slope': slope, 'intercept': intercept, 'exponent': -slope,
        'r2': 1-sse/syy if syy > 0 else None,
        'rmse': math.sqrt(sse/len(xs)),
    }


def _region_rows(rows, fit):
    n = len(rows)
    # Rank quantiles chosen before fitting, consistently across all conditions.
    cuts = [0, max(1, math.ceil(n*.05)), max(2, math.ceil(n*.5)), n]
    regions = []
    for name, start, end in zip(['Head (first 5%)', 'Middle (5–50%)', 'Tail (last 50%)'], cuts[:-1], cuts[1:]):
        subset = rows[min(start,n):min(end,n)]
        local = fit_log_log(subset)
        residuals = []
        if fit['slope'] is not None:
            residuals = [math.log(row['cf']) - (fit['intercept']+fit['slope']*math.log(row['rank'])) for row in subset]
        regions.append({
            'name': name, 'start': subset[0]['rank'] if subset else None,
            'end': subset[-1]['rank'] if subset else None, 'terms': len(subset),
            'occurrences': sum(row['cf'] for row in subset),
            'examples': subset[:8],
            'global_rmse': math.sqrt(sum(e*e for e in residuals)/len(residuals)) if residuals else None,
            **local,
        })
    return regions


def analyze(abstracts):
    abstracts = list(abstracts)
    cfs = {key: Counter() for key in CONDITIONS}
    dfs = {key: Counter() for key in CONDITIONS}
    forms = defaultdict(Counter)
    for abstract in abstracts:
        tokens = condition_tokens(abstract)
        for key, values in tokens.items():
            cfs[key].update(values)
            dfs[key].update(set(values))
        for original, stem in zip(tokens['C'], tokens['D']):
            forms[stem][original] += 1
    result = {}
    count = len(abstracts)
    for key, (label, description) in CONDITIONS.items():
        cf, df = cfs[key], dfs[key]
        rows = [
            {'rank': rank, 'term': term, 'cf': frequency, 'df': df[term],
             'idf': math.log(count/df[term]), 'coverage': 100*df[term]/count}
            for rank, (term, frequency) in enumerate(sorted(cf.items(), key=lambda item: (-item[1],item[0])), 1)
        ]
        fit = fit_log_log(rows)
        total_tokens = sum(cf.values())
        result[key] = {
            'key': key, 'label': label, 'description': description,
            'documents': count, 'tokens': total_tokens, 'vocabulary': len(rows),
            'avg_tokens': total_tokens/count if count else 0,
            'hapax': sum(row['cf']==1 for row in rows), 'fit': fit,
            'rows': rows, 'regions': _region_rows(rows, fit),
            'top_terms': [row['term'] for row in rows[:5]],
        }
    examples = [
        {'stem': stem, 'forms': ', '.join(sorted(originals)), 'cf': sum(originals.values())}
        for stem, originals in forms.items() if len(originals)>1
    ]
    examples.sort(key=lambda row: (-row['cf'], row['stem']))
    result['D']['examples'] = examples[:20]
    return result


def analyze_queryset(documents):
    records = [(pk, abstract) for pk, abstract in documents.order_by('pk').values_list('pk','abstract') if abstract.strip()]
    digest = hashlib.sha256()
    for pk, abstract in records:
        digest.update(f'{pk}\0{abstract}\0'.encode('utf-8'))
    key = 'zipf-punctuation-a-v3-' + digest.hexdigest()
    result = cache.get(key)
    if result is None:
        result = analyze(abstract for _,abstract in records)
        cache.set(key,result,300)
    return result


def chart_data(result):
    output = {}
    for key, data in result.items():
        rows = data['rows']
        n = len(rows)
        # Plot log-spaced points, always retaining the top 50 and endpoints.
        indexes = set(range(min(50,n)))
        if n > 1:
            indexes.update(round(math.exp(i*math.log(n)/750))-1 for i in range(751))
        points = [{'rank': rows[i]['rank'], 'cf': rows[i]['cf'], 'term': rows[i]['term']} for i in sorted(indexes) if 0<=i<n]
        output[key] = {**{name:data[name] for name in ['key','label','documents','tokens','vocabulary','fit','regions']}, 'points': points}
    return output
