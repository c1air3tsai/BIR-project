# =============================================================================
# indexing/views.py
# Project 5 網頁：/indexing/（主選單的「Indexing」，全部內容在同一頁）
#
#   ① 資料與前處理   ② Part 1 Inverted Index（即時查詢）   ③ BSBI   ④ SPIMI   ⑤ 效能比較   ⑥ 檢索測試
#
# 資料來源可以切換（網址參數 source=reuters / pubmed，sources.py），一次顯示一個來源的結果。
# 效能數字與中間結果讀 run_indexing_experiments 產生的 summary.json（Reuters 在 docs/project5_results/，
# PubMed 在 docs/project5_results/pubmed/）；查詢是即時的：Part 1 用記憶體裡的基準索引，⑥ 直接讀 BSBI / SPIMI
# 寫在磁碟上的最終索引。
# 網址參數：source 資料來源；a、b 兩個查詢詞（single-term 各查一次，再做 AND / OR）
# 點頁面上任何一個 docID → document() 回傳那篇文件的全文（JSON），查詢詞用螢光筆標出來，顯示在彈出視窗
# =============================================================================
import json
import re
from functools import lru_cache

from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils.html import escape

from reuters.corpus import documents_by_id
from search.text_processing import iter_token_matches

from . import sources
from .experiments import LABELS, MB, STAGES, results_dir, work_dir
from .inverted_index import DiskIndex, InvertedIndex, normalize
from .queries import DEFAULT_TERMS, EXAMPLE_QUERIES, QUERIES

# 作業 Part 1 範例表格裡的 4 個 terms，在語料上實際的 dictionary 與 postings
EXAMPLE_TERMS = ('information', 'retrieval', 'system', 'algorithm')
SHOW_DOCS = 20


# Part 1 的基準索引：每個來源第一次開頁面時建（Reuters 約 4 秒），之後留在記憶體
@lru_cache(maxsize=2)
def _baseline(source):
    return InvertedIndex(source=source)


# BSBI / SPIMI 寫在磁碟上的最終索引；還沒建好時回傳 None，而且不記住，建好之後重新整理就讀得到
_DISK = {}


def _disk(source, method):
    folder = work_dir(source) / 'full' / method / 'index'
    if (source, method) not in _DISK:
        if not (folder / 'dictionary.bin').is_file():
            return None
        _DISK[(source, method)] = DiskIndex(folder)
    return _DISK[(source, method)]


def _summary(source):
    path = results_dir(source) / 'summary.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def _source(request):
    source = request.GET.get('source', sources.DEFAULT_SOURCE)
    return source if source in sources.SOURCES and sources.available(source) else sources.DEFAULT_SOURCE


def _state(request, source):
    raw = [request.GET.get(key, default).strip()[:40] for key, default in zip('ab', DEFAULT_TERMS[source])]
    return {'a': raw[0], 'b': raw[1], 'terms': [normalize(term) for term in raw], 'source': source}


# 一組查詢在某個索引的結果：兩個 single-term、AND、OR
def _boolean(index, a, b):
    result = index.boolean(a, b)
    return {key: {'count': len(ids), 'ids': ids[:SHOW_DOCS], 'more': max(0, len(ids) - SHOW_DOCS), 'all': ids}
            for key, ids in result.items()}


def _part1(state):
    index = _baseline(state['source'])
    a, b = state['terms']
    result = _boolean(index, a, b)
    examples = [{'term': t, 'df': index.df(t), 'postings': index.postings(t)[:12], 'more': max(0, index.df(t) - 12)}
                for t in EXAMPLE_TERMS]
    vocabulary = index.vocabulary()
    return {'result': result, 'examples': examples, 'vocabulary_size': len(vocabulary),
            'vocabulary_head': vocabulary[:24], 'documents': index.documents}


# ⑥：同一組查詢分別查 BSBI 與 SPIMI 的最終索引，比較是否完全相同
def _check(state):
    indexes = {method: _disk(state['source'], method) for method in ('bsbi', 'spimi')}
    if not all(indexes.values()):
        return None
    a, b = state['terms']
    results = {method: index.boolean(a, b) for method, index in indexes.items()}
    rows = []
    for key, label in (('a', a), ('b', b), ('and', f'{a} AND {b}'), ('or', f'{a} OR {b}')):
        first, second = results['bsbi'][key], results['spimi'][key]
        rows.append({'type': 'single-term' if key in 'ab' else key.upper(), 'expression': label,
                     'bsbi': len(first), 'spimi': len(second), 'identical': list(first) == list(second),
                     'head': list(first[:10]), 'terms': label if key in 'ab' else f'{a} {b}'})
    return rows


def _megabytes(value):
    return value / MB


def _results(summary, source):
    if summary is None:
        return None
    methods = summary['methods']
    for method, row in methods.items():
        row['label'] = LABELS[method]
        row['stages'] = [{'key': key, 'label': label, 'seconds': row['times'][key]} for key, label in STAGES.items()]
        row['index_seconds'] = row['seconds'] - row['times']['preprocess']
        row['peak_mb'] = _megabytes(row['peak'])
        row['intermediate_mb'] = _megabytes(row['intermediate_bytes'])
        row['final_mb'] = _megabytes(row['final_bytes'])
        row['dictionary_kb'] = row['index_bytes']['dictionary'] / 1024
        row['postings_mb'] = _megabytes(row['index_bytes']['postings'])
        row['term_id_map_mb'] = _megabytes(row['term_id_map_bytes'])
        for block in row['blocks']:
            block['kb'] = block['bytes'] / 1024
            block['memory_mb'] = _megabytes(block['memory'])
            block['docs'] = block['last_doc'] - block['first_doc'] + 1
    for rows in (summary['sweep'], summary['scaling']):
        for row in rows:
            row['label'] = LABELS[row['method']]
            row['peak_mb'] = _megabytes(row['peak_bytes']) if row['peak_bytes'] is not None else None
            row['intermediate_mb'] = _megabytes(row['intermediate_bytes'])
            row['final_mb'] = _megabytes(row['final_bytes'])
            row['repeated'] = row['documents'] > summary['dataset']['documents']
            row['index_memory_mb'] = _megabytes(row['index_memory']) if row.get('index_memory') else None
    summary['retrieval_ok'] = all(r['identical'] for r in summary['retrieval'])
    summary['queries'] = _query_groups(summary['retrieval'], QUERIES[source])
    summary['example_queries'] = _query_groups(summary['retrieval'], EXAMPLE_QUERIES)
    summary['sweep_pairs'] = _paired(summary['sweep'], 'limit_mb')
    summary['scaling_pairs'] = _paired(summary['scaling'], 'documents')
    # 說明文字要引用某一列時用：results.scale.100000.bsbi.seconds、results.limit.1.spimi.blocks；
    # results.scale.full 是整個語料那一列（Reuters 21,578 篇、PubMed 20,000 篇）
    summary['scale'] = {str(int(p['x'])): p for p in summary['scaling_pairs']}
    summary['scale']['full'] = summary['scale'].get(str(summary['dataset']['documents']))
    summary['limit'] = {str(int(p['x'])): p for p in summary['sweep_pairs']}
    # SPIMI 一個 block 就放得下整份索引的最小上限（Reuters 是 16 MB）
    one_block = [p['x'] for p in summary['sweep_pairs'] if p.get('spimi') and p['spimi']['blocks'] == 1]
    summary['spimi_one_block'] = summary['limit'][str(int(min(one_block)))] if one_block else None
    for row in (summary['heaps'] or {}).get('estimates', []):
        row['term_id_map_mb'] = _megabytes(row['term_id_map_bytes'])
    first, second = methods['bsbi'], methods['spimi']
    summary['ratios'] = {'time': first['seconds'] / second['seconds'],
                         'index_time': first['index_seconds'] / second['index_seconds'],
                         'memory': first['peak'] / second['peak'], 'blocks': first['block_count'] / second['block_count'],
                         'sort': first['times']['sort'] / max(second['times']['sort'], 1e-9),
                         'entries': first['block_entries'] / second['block_entries']}
    return summary


# ⑥ 的表格：每組查詢的 4 列（兩個 single-term、AND、OR）；terms 是點 docID 時要標螢光筆的詞
def _query_groups(rows, queries):
    groups = []
    for name, a, b in queries:
        found = [dict(r, terms=r['expression'] if r['type'] == 'single' else f'{a} {b}')
                 for r in rows if r['query'] == name]
        if found:
            groups.append({'name': name, 'a': a, 'b': b, 'rows': found})
    return groups


# 同一個 x（記憶體上限或文件數）的 BSBI、SPIMI 放在同一列
def _paired(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], {'x': row[key]})[row['method']] = row
    return list(groups.values())


def page(request):
    source = _source(request)
    state = _state(request, source)
    summary = _results(_summary(source), source)
    context = {
        'page_title': 'Indexing · BioMed IR',
        'source': sources.get(source),
        'sources': [dict(info, available=sources.available(key)) for key, info in sources.SOURCES.items()],
        'state': state, 'part1': _part1(state), 'check': _check(state), 'results': summary,
        'baseline': summary['methods'].get('baseline') if summary else None,
        'bsbi': summary['methods']['bsbi'] if summary else None,
        'spimi': summary['methods']['spimi'] if summary else None,
        'samples': summary['samples'] if summary else None,
        'queries': QUERIES[source], 'example_queries': EXAMPLE_QUERIES,
    }
    return render(request, 'indexing/indexing.html', context)


def result_file(request, name):
    folder = results_dir(_source(request)).resolve()
    path = (folder / name).resolve()
    if path.suffix != '.png' or path.parent != folder or not path.is_file():
        raise Http404('Result not found')
    return FileResponse(open(path, 'rb'), content_type='image/png')


# ---------------------------------------------------------------------------
# 點 docID 的彈出視窗：/indexing/doc/<docID>/?source=reuters&terms=oil price
# ---------------------------------------------------------------------------

# 用和索引相同的 tokenizer 切出每個 token 的位置，token（轉小寫後）就是查詢詞的地方包 <mark>；
# 和索引的比對規則一致：oil 會標，oil-related、oils 不會（索引裡它們是不同的 term）。counts 順便數次數
def _mark(text, counts):
    parts, last = [], 0
    for match in iter_token_matches(text):
        token = text[match.start():match.end()]
        if token.casefold() in counts:
            counts[token.casefold()] += 1
            parts.append(f'{escape(text[last:match.start()])}<mark>{escape(token)}</mark>')
            last = match.end()
    parts.append(escape(text[last:]))
    return ''.join(parts)


# Reuters 原文每行約 60 字就斷行，新段落以「換行 + 縮排」開頭：切成段落，段落內的換行合併成空白
def _paragraphs(text):
    text = re.sub(r'\n[ \t]{2,}', '\n\n', (text or '').strip())
    return [re.sub(r'\s*\n\s*', ' ', part).strip() for part in re.split(r'\n\s*\n', text) if part.strip()]


# 兩個來源的文件 → 彈出視窗要的欄位：標題、段落、上方的小字（日期、主題 / PMID、期刊、年份）、原文連結
def _document(source, doc_id):
    if source == 'pubmed':
        record = sources.pubmed_documents().get(doc_id)
        if record is None:
            return None
        return {'title': record['title'] or '(no title)', 'paragraphs': record['abstract'],
                'meta': [f"PMID {record['pmid']}", record['journal'], record['year']],
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/", 'link': '在 PubMed 開啟'}
    doc = documents_by_id().get(doc_id)
    if doc is None:
        return None
    return {'title': doc['title'] or '(no title)', 'paragraphs': _paragraphs(doc['body']),
            'meta': [doc['date'], ', '.join(doc['topics'])], 'url': reverse('reuters:news', args=[doc_id]),
            'link': '在新聞頁開啟'}


def document(request, doc_id):
    source = _source(request)
    doc = _document(source, doc_id)
    if doc is None:
        raise Http404(f"Document not found in {sources.get(source)['label']}")
    # 查詢詞做和索引一樣的前處理（轉小寫、去 stopword）；最多 4 個，重複的只留一個
    terms = [t for t in dict.fromkeys(normalize(word) for word in request.GET.get('terms', '').split()[:4]) if t]
    counts = dict.fromkeys(terms, 0)
    title = _mark(doc['title'], counts)
    paragraphs = [_mark(part, counts) for part in doc['paragraphs']]
    return JsonResponse({'id': doc_id, 'meta': [m for m in doc['meta'] if m], 'title': title, 'paragraphs': paragraphs,
                         'counts': [{'term': t, 'count': n} for t, n in counts.items()],
                         'url': doc['url'], 'link': doc['link']})
