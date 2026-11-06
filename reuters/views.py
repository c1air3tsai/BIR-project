# =============================================================================
# reuters/views.py
# Project 4 網頁：/reuters/（主選單的「Reuters」，全部功能在同一頁）
#
#   ① 資料集概況（可切換 README 的三種切分）   ② Word2Vec 訓練（gensim ＋ 自製 numpy 示範）
#   ③ 詞層級分析（Top-5 鄰近詞）   ④ 新聞搜尋（單一方法 Top 10 ＋ A / B 並排比較）   ⑤ 文件層級評估
# 另有：
#   新聞列表 /reuters/news/：依主題（TOPICS 標籤）瀏覽 Reuters-21578 的全部新聞
#   新聞內頁 /reuters/news/<NEWID>/：全文、主題、所屬切分、查詢詞高亮、這篇在各方法的排名、TF-IDF 權重最高的詞
#
# 搜尋與鄰近詞是即時計算；評估要跑一段時間，改讀 run_reuters_experiments 產生的結果檔
# （docs/project4_results/<切分>/），所以網頁開啟很快。
#
# 網址參數：split 切分、q 查詢、method 方法、topic 目標主題（用來標示哪些結果相關）、word 鄰近詞查詢
# =============================================================================
import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.core.paginator import Paginator
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.urls import reverse
from django.utils.html import escape
from django.utils.safestring import mark_safe

from . import embedding, toy_word2vec
from .corpus import (DATASET_NAME, DEFAULT_SPLIT, SPLIT_PARTS, SPLITS, TOPIC_NAMES, TOPICS, cache_path,
                     document_text, documents_by_id, load_documents, split_counts, split_documents)
from .evaluation import METRICS
from .queries import DEFAULT_QUERY, EVALUATION_QUERIES, NEIGHBOR_WORDS, WORD_SUGGESTIONS
from .retrieval import CORE_METHODS, METHODS, load_index
from .text import preprocess

RESULTS_DIR = Path(settings.BASE_DIR) / 'docs' / 'project4_results'
# 主題顏色（與圖表一致，Okabe-Ito 色盲友善配色）
TOPIC_SWATCHES = {'earn': '#E69F00', 'acq': '#CC79A7', 'crude': '#D55E00', 'grain': '#009E73', 'trade': '#0072B2'}
NEWS_PER_PAGE = 25


def _state(request):
    split = request.GET.get('split', DEFAULT_SPLIT)
    method = request.GET.get('method', 'w2v_tfidf_sg')
    topic = request.GET.get('topic', 'crude')
    return {'split': split if split in SPLITS else DEFAULT_SPLIT,
            'q': request.GET.get('q', DEFAULT_QUERY).strip()[:200] or DEFAULT_QUERY,
            'method': method if method in METHODS else 'w2v_tfidf_sg',
            'topic': topic if topic in TOPICS or topic == '' else 'crude',
            'word': request.GET.get('word', 'oil').strip()[:40] or 'oil'}


def _hidden(state, own):
    return [(k, v) for k, v in state.items() if k not in own]


# 同一頁、改幾個參數的網址（例如換切分、換查詢），hash 是要跳回的段落
def _page_url(state, hash='', **changes):
    return '?' + urlencode({**state, **changes}) + (f'#{hash}' if hash else '')


def _news_url(newid, state, **changes):
    return reverse('reuters:news', args=[newid]) + '?' + urlencode({**state, **changes})


def _topic_items(topics):
    return [{'key': t, 'name': TOPIC_NAMES.get(t, '')} for t in topics]


# 把一段文字中命中查詢詞的字包上 <mark>（保留原本的空白與換行）
def _highlight(text, query_terms):
    parts = []
    for piece in re.split(r'(\s+)', text):
        normalized = preprocess(piece) if piece.strip() else []
        parts.append(f'<mark>{escape(piece)}</mark>' if normalized and normalized[0] in query_terms else escape(piece))
    return mark_safe(''.join(parts))


# Reuters 原文每行約 60 字就斷行，新段落以「換行 + 縮排」開頭：
# 段落之間留空行，段落內的換行合併成空白
def _paragraphs(text):
    text = re.sub(r'\n[ \t]{2,}', '\n\n', text.strip())
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', text)


# 內文前 260 字（換行合併成空白），命中查詢詞的地方包 <mark>
def _snippet(doc, query_terms, length=260):
    text = re.sub(r'\s+', ' ', doc['body'][:length]) + ('…' if len(doc['body']) > length else '')
    return _highlight(text, query_terms)


def _results(index, state, method, limit=10):
    terms = set(preprocess(state['q']))
    topic = state['topic']
    rows = []
    for rank, (row, score) in enumerate(index.ranking(state['q'], method, limit=limit), 1):
        doc = index.documents[row]
        rows.append({'rank': rank, 'score': score, 'id': doc['id'], 'date': doc['date'], 'title': doc['title'] or '(no title)',
                     'topics': doc['topics'], 'topic_items': _topic_items(doc['topics']),
                     'relevant': bool(topic) and topic in doc['topics'],
                     'snippet': _snippet(doc, terms), 'url': _news_url(doc['id'], state, method=method)})
    return rows


def _read_csv(name, split=None):
    path = (RESULTS_DIR / split / name) if split else (RESULTS_DIR / name)
    if not path.is_file():
        return []
    with open(path, encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


# 評估結果（實驗指令產生的 CSV），轉成表格要用的格式
def _evaluation(state):
    split = state['split']
    means = _read_csv('mean_metrics.csv', split)
    if not means:
        return None
    best = {k: max(float(r[k]) for r in means) for k in METRICS}
    mean_rows = [{'label': r['label'], 'method': r['method'], 'core': r['method'] in CORE_METHODS,
                  'cells': [{'value': float(r[k]), 'width': round(float(r[k]) * 100, 1),
                             'best': abs(float(r[k]) - best[k]) < 1e-9} for k in METRICS]} for r in means]
    by_type = _read_csv('metrics_by_type.csv', split)
    type_rows = [{'label': METHODS[m],
                  'keyword': next(float(r['ndcg10']) for r in by_type if r['method'] == m and r['type'] == 'keyword'),
                  'paraphrase': next(float(r['ndcg10']) for r in by_type if r['method'] == m and r['type'] == 'paraphrase')}
                 for m in METHODS]
    query_rows = []
    for r in _read_csv('query_comparison.csv', split):
        values = [float(r['ndcg_tfidf']), float(r['ndcg_cbow']), float(r['ndcg_sg'])]
        top, bottom = max(values), min(values)
        query_rows.append({**r, 'cells': [{'value': v, 'best': v == top,
                                           'alpha': round(0.06 + 0.74 * ((v - bottom) / (top - bottom) if top > bottom else 1), 3),
                                           'dark': (v - bottom) / (top - bottom) > 0.6 if top > bottom else True}
                                          for v in values],
                           'delta': float(r['delta_sg_minus_tfidf']),
                           'url': _page_url(state, 'search', q=r['text'], method='tfidf', topic=r['topic'])})
    counts = split_counts()[split]
    corpus_docs = {'train': counts['train'], 'train_unused': counts['train'] + counts['unused']}
    corpus = [{**r, 'label': METHODS[r['method']], 'ndcg10': float(r['ndcg10']), 'p10': float(r['p10']),
               'mrr': float(r['mrr']), 'docs': corpus_docs[r['w2v_corpus']],
               'corpus_label': embedding.CORPORA[r['w2v_corpus']]} for r in _read_csv('corpus_size.csv', split)]
    return {'means': mean_rows, 'types': type_rows, 'queries': query_rows, 'corpus': corpus,
            'metric_names': list(METRICS.values())}


# 三種切分的檢索結果比較表：每個方法一列，nDCG@10 / P@10 / MRR 各有三個切分的欄位
def _split_results(state):
    rows = _read_csv('split_comparison.csv')
    if not rows:
        return None
    splits = [s for s in SPLITS if any(r['split'] == s for r in rows)]
    keys = ('ndcg10', 'p10', 'mrr', 'ndcg_paraphrase')
    table = []
    for method in METHODS:
        cells = []
        for key in keys:
            for split in splits:
                match = next((r for r in rows if r['split'] == split and r['method'] == method), None)
                cells.append({'value': float(match[key]) if match else None, 'active': split == state['split']})
        table.append({'label': METHODS[method], 'core': method in CORE_METHODS, 'cells': cells})
    return {'splits': [{'key': s, 'label': SPLITS[s]['label'], 'active': s == state['split']} for s in splits],
            'groups': ['nDCG@10', 'P@10', 'MRR', 'paraphrase nDCG@10'], 'rows': table,
            'has_chart': (RESULTS_DIR / 'split_comparison.png').is_file()}


# 三種切分的篇數、測試集的主題分布（切分選單與比較表用）
@lru_cache(maxsize=1)
def _split_overview():
    counts = split_counts()
    rows = []
    for key, info in SPLITS.items():
        test = split_documents('test', key)
        rows.append({'key': key, **info, 'counts': counts[key],
                     'topic_counts': [sum(t in doc['topics'] for doc in test) for t in TOPICS],
                     'no_topics': sum(not doc['topics'] for doc in test)})
    return rows


# 評估查詢清單：依主題分組，附上每個主題在訓練集裡最有代表性的詞（topic_keywords.csv）
def _query_groups(state):
    keywords = {}
    for row in _read_csv('topic_keywords.csv', state['split']):
        keywords.setdefault(row['topic'], []).append(row['term'])
    groups = []
    for topic, label in TOPICS.items():
        queries = [{**q, 'url': _page_url(state, 'search', q=q['text'], topic=q['topic']),
                    'active': q['text'].casefold() == state['q'].casefold()}
                   for q in EVALUATION_QUERIES if q['topic'] == topic]
        groups.append({'topic': topic, 'label': label, 'keywords': keywords.get(topic, [])[:8], 'queries': queries})
    return groups


def _word_suggestions(state):
    return [{'label': label, 'words': [{'word': word, 'hint': hint, 'url': _page_url(state, 'words', word=word),
                                        'active': word == state['word'].casefold()} for word, hint in words]}
            for label, words in WORD_SUGGESTIONS]


# 互動散布圖的座標（run_reuters_experiments 產生）；沒有就回傳 None，頁面改顯示提示
def _projection(split):
    path = RESULTS_DIR / split / 'projection.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


@lru_cache(maxsize=1)
def _toy():
    return toy_word2vec.demo()


def page(request):
    state = _state(request)
    split = state['split']
    index = load_index(split, 'train')
    summary = index.summary()
    metas = {a: embedding.load(split, 'train', a)[1] for a in embedding.ARCHITECTURES}

    neighbor_table = [{'word': w, 'cbow': embedding.neighbors(w, split, 'cbow') or [],
                       'sg': embedding.neighbors(w, split, 'skipgram') or []} for w in NEIGHBOR_WORDS]
    word_lookup = {'word': state['word'].casefold(),
                   'cbow': embedding.neighbors(state['word'], split, 'cbow', topn=10),
                   'sg': embedding.neighbors(state['word'], split, 'skipgram', topn=10)}

    comparison = [{'method': m, 'label': METHODS[m], 'rows': _results(index, state, m)} for m in CORE_METHODS]
    for column in comparison:
        column['hits'] = sum(r['relevant'] for r in column['rows'])

    overview = _split_overview()
    toy = _toy()
    topic_counts = summary['topic_counts']
    context = {
        'page_title': 'Reuters 檢索 · BioMed IR',
        'state': state, 'hidden_search': _hidden(state, ('q', 'method', 'topic')),
        'hidden_word': _hidden(state, ('word',)),
        'split_info': SPLITS[split], 'counts': split_counts()[split],
        'split_options': [{**row, 'url': _page_url(state, 'dataset', split=row['key']), 'active': row['key'] == split}
                          for row in overview],
        'topic_keys': list(TOPICS),
        'summary': summary, 'topics': TOPICS,
        'topic_rows': [{'key': k, 'label': v, 'count': topic_counts[k]} for k, v in TOPICS.items()],
        'relevant_range': (min(topic_counts.values()), max(topic_counts.values())),
        'metas': [{'key': a, 'label': embedding.ARCHITECTURES[a], **m, 'final_loss': m['epoch_losses'][-1]}
                  for a, m in metas.items()],
        'params': embedding.PARAMS,
        'toy': {'vocabulary': toy['vocabulary'], 'pairs': toy['pairs'], 'example': toy['example'],
                'sentences': toy['sentences'][:6],
                'rows': [{'arch': embedding.ARCHITECTURES[a], 'first': toy[a]['losses'][0], 'last': toy[a]['losses'][-1],
                          'neighbors': toy[a]['neighbors']} for a in ('cbow', 'skipgram')]},
        'neighbor_table': neighbor_table, 'word_lookup': word_lookup, 'word_groups': _word_suggestions(state),
        'methods': [{'key': k, 'label': v} for k, v in METHODS.items()],
        'method_label': METHODS[state['method']],
        'query_terms': index.query_terms(state['q']),
        'results': _results(index, state, state['method']),
        'comparison': comparison, 'query_groups': _query_groups(state),
        'evaluation': _evaluation(state), 'split_results': _split_results(state), 'eval_queries': EVALUATION_QUERIES,
        'has_results': (RESULTS_DIR / split / 'metrics_comparison.png').is_file(),
        'has_toy_chart': (RESULTS_DIR / 'toy_word2vec_loss.png').is_file(),
        'corpus_ready': cache_path().is_file(),
        'projection': _projection(split), 'topic_colors': TOPIC_SWATCHES,
        'news_list_url': reverse('reuters:news_list') + '?' + urlencode({'split': split}),
    }
    return render(request, 'reuters/reuters.html', context)


# 新聞列表的主題篩選：'' = 全部、'_none' = 沒有 TOPICS 標籤、其他 = 某個 TOPICS 標籤
def _filter_by_topic(docs, topic):
    if topic == '_none':
        return [doc for doc in docs if not doc['topics']]
    if topic:
        return [doc for doc in docs if topic in doc['topics']]
    return docs


# 新聞列表：依切分的哪一部分（全部 / 訓練集 / 測試集 / 未使用）與 TOPICS 主題瀏覽 Reuters-21578
def news_list(request):
    split = request.GET.get('split', DEFAULT_SPLIT)
    split = split if split in SPLITS else DEFAULT_SPLIT
    part = request.GET.get('part', 'all')
    part = part if part in SPLIT_PARTS or part == 'all' else 'all'
    title = request.GET.get('title', '').strip()[:80]
    docs = load_documents() if part == 'all' else split_documents(part, split)
    if title:
        needle = title.casefold()
        docs = [doc for doc in docs if needle in doc['title'].casefold()]

    # 篩選選單上的篇數：在目前的切分 / 部分 / 標題條件下，各主題有幾篇
    topic_totals = {}
    for doc in docs:
        for t in doc['topics']:
            topic_totals[t] = topic_totals.get(t, 0) + 1
    topic = request.GET.get('topic', '')
    topic = topic if topic in topic_totals or topic in TOPIC_NAMES or topic == '_none' else ''
    filtered = _filter_by_topic(docs, topic)
    page_obj = Paginator(filtered, NEWS_PER_PAGE).get_page(request.GET.get('page'))
    rows = [{'id': doc['id'], 'date': doc['date'], 'title': doc['title'] or '(no title)',
             'topic_items': _topic_items(doc['topics']), 'part': SPLIT_PARTS[doc['splits'][split]],
             'snippet': re.sub(r'\s+', ' ', doc['body'][:220]) + ('…' if len(doc['body']) > 220 else ''),
             'url': reverse('reuters:news', args=[doc['id']]) + '?' + urlencode({'split': split})}
            for doc in page_obj]

    base = {'split': split, 'part': part, 'title': title}
    others = sorted((t for t in topic_totals if t not in TOPICS), key=lambda t: (-topic_totals[t], t))
    return render(request, 'reuters/news_list.html', {
        'page_title': f'{DATASET_NAME} 新聞列表 · Reuters', 'dataset': DATASET_NAME,
        'split': split, 'split_info': SPLITS[split], 'part': part, 'title': title, 'topic': topic,
        'topic_name': TOPIC_NAMES.get(topic, '') if topic not in ('', '_none') else '',
        'total': len(docs), 'count': len(filtered), 'page_obj': page_obj, 'rows': rows,
        'splits': [{'key': k, 'label': v['label']} for k, v in SPLITS.items()],
        'parts': [('all', f'全部 {len(load_documents()):,} 篇')] + [(k, v) for k, v in SPLIT_PARTS.items()],
        'main_topics': [{'key': t, 'name': TOPIC_NAMES[t], 'count': topic_totals.get(t, 0),
                         'url': '?' + urlencode({**base, 'topic': t})} for t in TOPICS],
        'other_topics': [{'key': t, 'name': TOPIC_NAMES.get(t, ''), 'count': topic_totals[t]} for t in others],
        'no_topic_count': sum(not doc['topics'] for doc in docs),
        'all_url': '?' + urlencode(base), 'none_url': '?' + urlencode({**base, 'topic': '_none'}),
        'page_url': '?' + urlencode({**base, 'topic': topic}) + '&page=',
        'search_url': reverse('reuters:page') + '?' + urlencode({'split': split}) + '#search',
    })


# 新聞內頁：全文 + 查詢詞高亮；這篇在三種切分各屬於哪一部分；
# 如果它在目前切分的測試集裡、而且有帶查詢，顯示它在 A / B 三種方法的排名
def news_detail(request, newid):
    doc = documents_by_id().get(newid)
    if doc is None:
        raise Http404('News not found in Reuters-21578')
    state = _state(request)
    index = load_index(state['split'], 'train')
    row = index.row_of.get(newid)
    has_query = bool(request.GET.get('q', '').strip())
    terms = set(preprocess(state['q'])) if has_query else set()
    doc_terms = index.doc_terms[row] if row is not None else preprocess(document_text(doc))
    ranks = ([{'label': METHODS[m], 'rank': index.rank_of(state['q'], m, row)} for m in CORE_METHODS]
             if has_query and row is not None else [])
    return render(request, 'reuters/news.html', {
        'page_title': f"{doc['title'] or 'Reuters'} · Reuters", 'doc': doc, 'state': state, 'has_query': has_query,
        'split_info': SPLITS[state['split']], 'in_test': row is not None, 'dataset': DATASET_NAME,
        'part': SPLIT_PARTS[doc['splits'][state['split']]],
        'memberships': [{'label': info['label'], 'part': SPLIT_PARTS[doc['splits'][key]], 'active': key == state['split'],
                         'test': doc['splits'][key] == 'test'} for key, info in SPLITS.items()],
        'topic_items': _topic_items(doc['topics']),
        'title': _highlight(doc['title'] or '(no title)', terms), 'body': _highlight(_paragraphs(doc['body']), terms),
        'relevant': bool(state['topic']) and state['topic'] in doc['topics'],
        'ranks': ranks, 'top_terms': index.top_terms(doc_terms), 'length': len(doc_terms),
        'back_url': (reverse('reuters:page') + _page_url(state, 'search')) if has_query
                    else reverse('reuters:news_list') + '?' + urlencode({'split': state['split']}),
        'word_url': reverse('reuters:page') + '?' + urlencode({'split': state['split']}) + '&word=',
    })


# 顯示實驗指令產生的 PNG：docs/project4_results/ 最上層（三種切分共用）或 <切分>/ 裡的 .png
def result_file(request, name, split=None):
    if split is not None and split not in SPLITS:
        raise Http404('Result not found')
    folder = (RESULTS_DIR / split if split else RESULTS_DIR).resolve()
    path = (folder / name).resolve()
    if path.suffix != '.png' or path.parent != folder or not path.is_file():
        raise Http404('Result not found')
    return FileResponse(open(path, 'rb'), content_type='image/png')
