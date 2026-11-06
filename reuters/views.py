# =============================================================================
# reuters/views.py
# Project 4 網頁：/reuters/（主選單的「Reuters」，全部功能在同一頁）
#
#   ① 資料集概況   ② Word2Vec 訓練（gensim ＋ 自製 numpy 示範）   ③ 詞層級分析（Top-5 鄰近詞）
#   ④ 新聞搜尋（單一方法 Top 10 ＋ A / B 並排比較）   ⑤ 文件層級評估
# 另有新聞內頁 /reuters/news/<NEWID>/：全文、查詢詞高亮、這篇在各方法的排名、TF-IDF 權重最高的詞
#
# 搜尋與鄰近詞是即時計算；評估要跑約 10 秒，改讀 run_reuters_experiments 產生的結果檔
# （docs/project4_results/），所以網頁開啟很快。
#
# 網址參數：q 查詢、method 方法、topic 目標主題（用來標示哪些結果相關）、word 鄰近詞查詢
# =============================================================================
import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.utils.html import escape
from django.utils.safestring import mark_safe

from . import embedding, toy_word2vec
from .corpus import TOPICS, cache_path
from .evaluation import METRICS
from .queries import DEMO_QUERIES, EVALUATION_QUERIES, NEIGHBOR_WORDS
from .retrieval import CORE_METHODS, METHODS, load_index
from .text import preprocess

RESULTS_DIR = Path(settings.BASE_DIR) / 'docs' / 'project4_results'
DEFAULT_QUERY = DEMO_QUERIES[0]
# 主題顏色（與圖表一致，Okabe-Ito 色盲友善配色）
TOPIC_SWATCHES = {'earn': '#E69F00', 'acq': '#CC79A7', 'crude': '#D55E00', 'grain': '#009E73', 'trade': '#0072B2'}


def _state(request):
    method = request.GET.get('method', 'w2v_tfidf_sg')
    topic = request.GET.get('topic', 'crude')
    return {'q': request.GET.get('q', DEFAULT_QUERY).strip()[:200] or DEFAULT_QUERY,
            'method': method if method in METHODS else 'w2v_tfidf_sg',
            'topic': topic if topic in TOPICS or topic == '' else 'crude',
            'word': request.GET.get('word', 'oil').strip()[:40] or 'oil'}


def _hidden(state, own):
    return [(k, v) for k, v in state.items() if k not in own]


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


def _results(index, query, method, topic, limit=10):
    terms = set(preprocess(query))
    rows = []
    for rank, (row, score) in enumerate(index.ranking(query, method, limit=limit), 1):
        doc = index.documents[row]
        rows.append({'rank': rank, 'score': score, 'id': doc['id'], 'date': doc['date'], 'title': doc['title'] or '(no title)',
                     'topics': doc['topics'], 'relevant': bool(topic) and topic in doc['topics'],
                     'snippet': _snippet(doc, terms)})
    return rows


def _read_csv(name):
    path = RESULTS_DIR / name
    if not path.is_file():
        return []
    with open(path, encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


# 評估結果（實驗指令產生的 CSV），轉成表格要用的格式
def _evaluation():
    means = _read_csv('mean_metrics.csv')
    if not means:
        return None
    best = {k: max(float(r[k]) for r in means) for k in METRICS}
    mean_rows = [{'label': r['label'], 'method': r['method'], 'core': r['method'] in CORE_METHODS,
                  'cells': [{'value': float(r[k]), 'width': round(float(r[k]) * 100, 1),
                             'best': abs(float(r[k]) - best[k]) < 1e-9} for k in METRICS]} for r in means]
    by_type = _read_csv('metrics_by_type.csv')
    type_rows = [{'label': METHODS[m],
                  'keyword': next(float(r['ndcg10']) for r in by_type if r['method'] == m and r['type'] == 'keyword'),
                  'paraphrase': next(float(r['ndcg10']) for r in by_type if r['method'] == m and r['type'] == 'paraphrase')}
                 for m in METHODS]
    comparison = _read_csv('query_comparison.csv')
    query_rows = []
    for r in comparison:
        values = [float(r['ndcg_tfidf']), float(r['ndcg_cbow']), float(r['ndcg_sg'])]
        top, bottom = max(values), min(values)
        query_rows.append({**r, 'cells': [{'value': v, 'best': v == top,
                                           'alpha': round(0.06 + 0.74 * ((v - bottom) / (top - bottom) if top > bottom else 1), 3),
                                           'dark': (v - bottom) / (top - bottom) > 0.6 if top > bottom else True}
                                          for v in values],
                           'delta': float(r['delta_sg_minus_tfidf'])})
    corpus = [{**r, 'label': METHODS[r['method']], 'ndcg10': float(r['ndcg10']), 'p10': float(r['p10']),
               'mrr': float(r['mrr'])} for r in _read_csv('corpus_size.csv')]
    return {'means': mean_rows, 'types': type_rows, 'queries': query_rows, 'corpus': corpus,
            'metric_names': list(METRICS.values())}


# 互動散布圖的座標（run_reuters_experiments 產生）；沒有就回傳 None，頁面改顯示提示
def _projection():
    path = RESULTS_DIR / 'projection.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


@lru_cache(maxsize=1)
def _toy():
    return toy_word2vec.demo()


def page(request):
    state = _state(request)
    index = load_index('train')
    metas = {a: embedding.load('train', a)[1] for a in embedding.ARCHITECTURES}

    neighbor_table = [{'word': w, 'cbow': embedding.neighbors(w, 'train', 'cbow') or [],
                       'sg': embedding.neighbors(w, 'train', 'skipgram') or []} for w in NEIGHBOR_WORDS]
    word_lookup = {'word': state['word'].casefold(),
                   'cbow': embedding.neighbors(state['word'], 'train', 'cbow', topn=10),
                   'sg': embedding.neighbors(state['word'], 'train', 'skipgram', topn=10)}

    comparison = [{'method': m, 'label': METHODS[m], 'rows': _results(index, state['q'], m, state['topic'])}
                  for m in CORE_METHODS]
    for column in comparison:
        column['hits'] = sum(r['relevant'] for r in column['rows'])

    toy = _toy()
    context = {
        'page_title': 'Reuters 檢索 · BioMed IR',
        'state': state, 'hidden_search': _hidden(state, ('q', 'method', 'topic')),
        'hidden_word': _hidden(state, ('word',)),
        'summary': index.summary(), 'topics': TOPICS,
        'topic_rows': [{'key': k, 'label': v, 'count': index.summary()['topic_counts'][k]} for k, v in TOPICS.items()],
        'metas': [{'key': a, 'label': embedding.ARCHITECTURES[a], **m, 'final_loss': m['epoch_losses'][-1]}
                  for a, m in metas.items()],
        'params': embedding.PARAMS,
        'toy': {'vocabulary': toy['vocabulary'], 'pairs': toy['pairs'], 'example': toy['example'],
                'sentences': toy['sentences'][:6],
                'rows': [{'arch': embedding.ARCHITECTURES[a], 'first': toy[a]['losses'][0], 'last': toy[a]['losses'][-1],
                          'neighbors': toy[a]['neighbors']} for a in ('cbow', 'skipgram')]},
        'neighbor_table': neighbor_table, 'word_lookup': word_lookup,
        'methods': [{'key': k, 'label': v} for k, v in METHODS.items()],
        'method_label': METHODS[state['method']],
        'query_terms': index.query_terms(state['q']),
        'results': _results(index, state['q'], state['method'], state['topic']),
        'comparison': comparison, 'demo_queries': DEMO_QUERIES,
        'evaluation': _evaluation(), 'eval_queries': EVALUATION_QUERIES,
        'has_results': (RESULTS_DIR / 'metrics_comparison.png').is_file(),
        'corpus_ready': cache_path().is_file(),
        'projection': _projection(), 'topic_colors': TOPIC_SWATCHES,
    }
    return render(request, 'reuters/reuters.html', context)


# 新聞內頁：全文 + 查詢詞高亮；有帶查詢時，顯示這篇在 A / B 三種方法的排名
def news_detail(request, newid):
    index = load_index('train')
    row = index.row_of.get(newid)
    if row is None:
        raise Http404('News not found in the ModApte test set')
    state = _state(request)
    has_query = bool(request.GET.get('q', '').strip())
    doc = index.documents[row]
    terms = set(preprocess(state['q'])) if has_query else set()
    ranks = [{'label': METHODS[m], 'rank': index.rank_of(state['q'], m, row)} for m in CORE_METHODS] if has_query else []
    return render(request, 'reuters/news.html', {
        'page_title': f"{doc['title'] or 'Reuters'} · Reuters", 'doc': doc, 'state': state, 'has_query': has_query,
        'title': _highlight(doc['title'] or '(no title)', terms), 'body': _highlight(_paragraphs(doc['body']), terms),
        'relevant': bool(state['topic']) and state['topic'] in doc['topics'],
        'ranks': ranks, 'top_terms': index.top_terms(row), 'length': int(index.lengths[row]),
        'topics': TOPICS,
    })


# 顯示實驗指令產生的 PNG（只允許 docs/project4_results/ 裡的 .png）
def result_file(request, name):
    path = (RESULTS_DIR / name).resolve()
    if path.suffix != '.png' or path.parent != RESULTS_DIR.resolve() or not path.is_file():
        raise Http404('Result not found')
    return FileResponse(open(path, 'rb'), content_type='image/png')
