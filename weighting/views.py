# =============================================================================
# weighting/views.py
# Project 3 網頁：/analysis/term-weighting/（全部功能在同一頁，依作業任務分段）
#
#   ① 詞項統計（任務一）  ② 權重方法（任務二）  ③ 搜尋 Top 10（任務三）
#   ④ 比較模式（任務三）  ⑤ 評估（任務五）
#   對照實驗 RQ1/RQ2/RQ4（任務四）的表格已從網頁拿掉，由 run_weighting_experiments 輸出到 docs/
#
# 頁面用一般的 GET 表單（不需要 JavaScript），網址參數：
#   stop=1/0   是否移除 stopword        stem=1/0   是否做 Porter stemming
#   q          查詢                      method     權重方法代號（見 schemes.METHODS）
#   l2=1/0     是否 L2 normalization     term       詞項查詢框的輸入
#
# 另外兩個網址：
#   export/        下載目前設定的評估結果 CSV（每個查詢 × 每種方法）
#   results/<名>   顯示 run_weighting_experiments 產生的 PNG 圖
# =============================================================================
import csv
import io
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import render
from django.utils.html import escape
from django.utils.safestring import mark_safe

from . import experiments
from .evaluation import METRICS, evaluate_all
from .index import config_from_flags, load_index
from .preprocessing import term_spans
from .queries import DEMO_QUERIES  # 恢復對照實驗區塊時改回 import DEMO_QUERIES, DEMO_TERMS
from .schemes import FORMULAS, METHODS, REQUIRED_METHODS

RESULTS_DIR = Path(settings.BASE_DIR) / 'docs' / 'project3_results'
DEFAULT_QUERY = DEMO_QUERIES[0]


# 讀取並檢查網址參數，不合法的值改用預設值
def _state(request):
    method = request.GET.get('method', 'logtfidf')
    return {
        'stop': '0' if request.GET.get('stop') == '0' else '1',
        'stem': '1' if request.GET.get('stem') == '1' else '0',
        'q': request.GET.get('q', DEFAULT_QUERY).strip()[:300] or DEFAULT_QUERY,
        'method': method if method in METHODS else 'logtfidf',
        'l2': '0' if request.GET.get('l2') == '0' else '1',
        'term': request.GET.get('term', 'semaglutide').strip()[:60],
    }


# 每個表單只送出自己的欄位，其他目前狀態用 hidden input 帶著走
def _hidden(state, own_fields):
    return [(key, value) for key, value in state.items() if key not in own_fields]


# 摘要片段：從第一個命中詞前面一點開始取約 360 字，命中查詢詞的位置包上 <mark>
def highlight_snippet(text, query_terms, config, length=360):
    text = text or ''
    hits = [(start, end) for start, end, term in term_spans(text, config) if term in query_terms]
    begin = max(0, hits[0][0] - 80) if hits else 0
    finish = min(len(text), begin + length)
    parts, cursor = [], begin
    for start, end in hits:
        if start < begin or end > finish:
            continue
        parts.append(escape(text[cursor:start]))
        parts.append(f'<mark>{escape(text[start:end])}</mark>')
        cursor = end
    parts.append(escape(text[cursor:finish]))
    prefix = '… ' if begin > 0 else ''
    suffix = ' …' if finish < len(text) else ''
    return mark_safe(prefix + ''.join(parts) + suffix)


# 搜尋結果：加上標題、PMID、摘要高亮等顯示資料
def _search_rows(index, result):
    documents = index.documents()
    query_terms = set(result['terms'])
    rows = []
    for rank, (row, score) in enumerate(result['ranked'], 1):
        document = documents[row]
        rows.append({'rank': rank, 'score': score, 'pmid': index.pmids[row], 'doc_id': index.doc_ids[row],
                     'title': document['title'], 'year': document['publication_year'],
                     'length': int(index.doc_lengths[row]),
                     'matched': [term for term in result['terms'] if index.counts[row, index.term_to_col[term]] > 0],
                     'snippet': highlight_snippet(document['abstract'], query_terms, index.config)})
    return rows


# 評估平均表：加上每格長條寬度（%），並標出每個指標的最高值
def _evaluation_table(evaluation):
    best = {key: max(row[key] for row in evaluation['means']) for key in METRICS}
    rows = []
    for row in evaluation['means']:
        rows.append({**row, 'cells': [{'value': row[key], 'width': round(row[key] * 100, 1),
                                       'best': abs(row[key] - best[key]) < 1e-9} for key in METRICS]})
    return rows


# 逐查詢 nDCG@10（目前選的 L2 設定），讓使用者看到不同查詢的差異
def _per_query_ndcg(evaluation, use_l2):
    table = []
    for query in evaluation['queries']:
        values = {row['method']: row['ndcg10'] for row in evaluation['per_query']
                  if row['query_id'] == query['id'] and row['use_l2'] == use_l2}
        best = max(values.values()) if values else 0
        worst = min(values.values()) if values else 0
        cells = []
        for method in METHODS:
            value = values.get(method, 0)
            # 同一列（同一個查詢）內縮放到 0–1：顏色越深代表在這個查詢裡排名品質越好
            shade = (value - worst) / (best - worst) if best > worst else 1.0
            cells.append({'value': value, 'best': value == best and best > 0,
                          'alpha': round(0.06 + 0.74 * shade, 3), 'dark': shade > 0.6})
        table.append({**query, 'cells': cells})
    return table


def page(request):
    state = _state(request)
    config = config_from_flags(state['stop'], state['stem'])
    index = load_index(config)
    use_l2 = state['l2'] == '1'

    # ① 詞項統計
    summary = index.summary()
    histogram = index.length_histogram(14)
    peak = max((bin_['count'] for bin_ in histogram), default=1) or 1
    for bin_ in histogram:
        bin_['height'] = round(bin_['count'] / peak * 100, 1)

    # ③ 搜尋 與 ④ 比較
    result = index.search(state['q'], state['method'], use_l2, limit=10)
    comparison = experiments.rank_comparison(index, state['q'], use_l2, baseline=state['method'])

    # ⑥ 評估
    evaluation = evaluate_all(index)

    context = {
        'analysis_tab': 'term_weighting', 'page_title': 'Term Weighting · BioMed IR',
        'state': state, 'config': config, 'use_l2': use_l2,
        'hidden_settings': _hidden(state, ('stop', 'stem')),
        'hidden_search': _hidden(state, ('q', 'method', 'l2')),
        'hidden_term': _hidden(state, ('term',)),
        'summary': summary, 'histogram': histogram,
        'top_df': index.top_df_terms(20),
        'term_info': index.term_info(state['term'], top_documents=8) if state['term'] else None,
        'methods': [{'key': key, 'label': label, 'formula': FORMULAS[key], 'required': key in REQUIRED_METHODS}
                    for key, label in METHODS.items()],
        'method_label': METHODS[state['method']],
        'demo_queries': DEMO_QUERIES,
        'result': result, 'results': _search_rows(index, result),
        'comparison': comparison,
        # 對照實驗（RQ1 / RQ2 / RQ4）區塊已從網頁註解掉，需要時取消下面五行的註解
        # 'tf_scaling': experiments.tf_scaling_table(),
        # 'term_weights': experiments.term_weight_demo(index, DEMO_TERMS),
        # 'idf_groups': experiments.idf_high_low(index),
        # 'length_demo': experiments.length_demo(index, state['q'], state['method']),
        # 'length_bias': experiments.length_bias_table(index),
        'metric_names': list(METRICS.values()),
        'evaluation_rows': _evaluation_table(evaluation),
        'evaluation_queries': _per_query_ndcg(evaluation, use_l2),
        'method_labels': list(METHODS.values()),
        'has_png': (RESULTS_DIR / 'metrics_comparison.png').is_file(),
    }
    return render(request, 'weighting/term_weighting.html', context)


# 下載目前前處理設定下的評估 CSV（每個查詢 × 每種方法 × 有無 L2，最後附平均）
def export_csv(request):
    state = _state(request)
    index = load_index(config_from_flags(state['stop'], state['stem']))
    evaluation = evaluate_all(index)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['query_id', 'configuration', 'method', 'use_l2', *METRICS.values()])
    for row in evaluation['per_query']:
        writer.writerow([row['query_id'], row['label'], row['method'], int(row['use_l2']),
                         *[f'{row[key]:.4f}' for key in METRICS]])
    for row in evaluation['means']:
        writer.writerow(['MEAN', row['label'], row['method'], int(row['use_l2']),
                         *[f'{row[key]:.4f}' for key in METRICS]])
    response = HttpResponse('﻿' + buffer.getvalue(), content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="weighting_evaluation_{index.config.key}.csv"'
    return response


# 顯示實驗指令產生的 PNG 圖（只允許 docs/project3_results/ 裡的 .png 檔）
def result_file(request, name):
    path = (RESULTS_DIR / name).resolve()
    if path.suffix != '.png' or path.parent != RESULTS_DIR.resolve() or not path.is_file():
        raise Http404('Result not found')
    return FileResponse(open(path, 'rb'), content_type='image/png')
