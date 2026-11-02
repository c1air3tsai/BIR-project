# =============================================================================
# weighting/experiments.py
# Project 3 任務四（對照實驗）＋ 任務五（評估輸出）
#
# 控制變因：資料集（1000 篇 GLP-1 摘要）、前處理（DEFAULT_CONFIG）、查詢集（queries.py）固定，
#           只改變 weighting 方法與有無 L2 normalization。
#
# 這裡的函式有兩種用途：
#   1. 網頁（weighting/views.py）即時呼叫 *_demo() 產生表格
#   2. python manage.py run_weighting_experiments 呼叫 run_all()，把所有 CSV / PNG 存到
#      docs/project3_results/，報告 docs/project3_report.md 引用的數字都來自這些檔案
# =============================================================================
import csv
import json
from pathlib import Path

import numpy as np

from . import schemes
from .evaluation import METRICS, config_label, configurations, evaluate_all, relevance_grades
from .queries import DEMO_TERMS, EVALUATION_QUERIES, LENGTH_DEMO_QUERY
from .schemes import METHODS, REQUIRED_METHODS

# 圖表顏色：沿用網站 Zipf 圖的 Okabe-Ito 色盲友善配色
COLOR_NO_L2 = '#D55E00'   # 朱紅：無 L2
COLOR_L2 = '#0072B2'      # 藍：有 L2（cosine）
INK = '#243746'
MUTED = '#687985'
GRID = '#dbe3e9'


# ---------------------------------------------------------------------------
# RQ1：Raw TF 與 Log TF 的差異
# ---------------------------------------------------------------------------

# tf 從 1 增加到 20 時，raw TF 與 log TF 權重的變化（看出 log 的「邊際遞減」）
def tf_scaling_table(values=(1, 2, 3, 5, 10, 20)):
    return [{'tf': tf, 'raw_tf': float(tf), 'log_tf': 1 + np.log10(tf),
             'raw_ratio': tf / 1.0, 'log_ratio': (1 + np.log10(tf)) / 1.0} for tf in values]


# 挑幾個詞，各取「tf 最高的文件」與「tf = 1 的文件」，列出同一詞在不同權重下的數值
def term_weight_demo(index, terms=DEMO_TERMS):
    rows = []
    for raw in terms:
        info = index.term_info(raw, top_documents=1)
        if not info.get('found'):
            continue
        col = index.term_to_col[info['term']]
        column = index.counts.getcol(col).tocoo()
        picks = []
        if len(column.data):
            # tf 最高的文件
            best = int(np.argmax(column.data))
            picks.append((int(column.row[best]), int(column.data[best])))
            # tf = 1 的文件（若存在且不是同一篇）
            ones = np.flatnonzero(column.data == 1)
            if len(ones) and int(column.row[ones[0]]) != picks[0][0]:
                picks.append((int(column.row[ones[0]]), 1))
        idf = float(index.idf[col])
        for row, tf in picks:
            log_tf = 1 + np.log10(tf)
            rows.append({'term': info['term'], 'df': info['df'], 'idf': idf, 'pmid': index.pmids[row],
                         'doc_length': int(index.doc_lengths[row]), 'tf': tf,
                         'raw_tf': float(tf), 'log_tf': log_tf, 'tfidf': tf * idf, 'logtfidf': log_tf * idf})
    return rows


# ---------------------------------------------------------------------------
# RQ2：高 DF 與低 DF 詞的 IDF
# ---------------------------------------------------------------------------

# 高 DF：出現在最多文件的詞；低 DF：df 介於 2–5、但總次數 cf 最高的詞（少數文件反覆討論的專門詞）
def idf_high_low(index, n=10):
    high = index.top_df_terms(n)
    candidates = np.flatnonzero((index.df >= 2) & (index.df <= 5))
    order = candidates[np.lexsort((candidates, -index.cf[candidates]))][:n]
    low = [index.term_row(col) for col in order]
    return {'high': high, 'low': low}


# ---------------------------------------------------------------------------
# RQ4：L2 normalization 與文件長度
# ---------------------------------------------------------------------------

# 同一查詢、同一方法，比較「無 L2（內積）」與「有 L2（cosine）」的 Top 10 與文件長度
def length_demo(index, query=LENGTH_DEMO_QUERY, method='tfidf'):
    documents = index.documents()
    sides = {}
    for use_l2 in (False, True):
        ranked = index.search(query, method, use_l2, limit=10)['ranked']
        sides[use_l2] = [{'rank': rank, 'row': row, 'pmid': index.pmids[row], 'score': score,
                          'title': documents[row]['title'], 'length': int(index.doc_lengths[row])}
                         for rank, (row, score) in enumerate(ranked, 1)]
    raw_rows = [item['row'] for item in sides[False]]
    l2_rows = [item['row'] for item in sides[True]]
    # 在另一邊的名次（不在對方 Top 10 時為 None），方便標示排名變化
    for item in sides[False]:
        item['other_rank'] = l2_rows.index(item['row']) + 1 if item['row'] in l2_rows else None
    for item in sides[True]:
        item['other_rank'] = raw_rows.index(item['row']) + 1 if item['row'] in raw_rows else None
    mean = lambda items: float(np.mean([item['length'] for item in items])) if items else 0.0
    return {'query': query, 'method': method, 'no_l2': sides[False], 'l2': sides[True],
            'mean_length_no_l2': mean(sides[False]), 'mean_length_l2': mean(sides[True]),
            'corpus_mean_length': index.avg_length, 'overlap': len(set(raw_rows) & set(l2_rows))}


# 對所有評估查詢：每個組合 Top 10 文件的平均長度，以及有無 L2 的 Top 10 重疊數
def length_bias_table(index, queries=EVALUATION_QUERIES):
    rows = []
    for method in METHODS:
        lengths = {False: [], True: []}
        overlaps = []
        for query in queries:
            ranked = {use_l2: index.ranking(query['text'], method, use_l2, limit=10) for use_l2 in (False, True)}
            for use_l2 in (False, True):
                lengths[use_l2].extend(int(index.doc_lengths[row]) for row in ranked[use_l2])
            overlaps.append(len(set(ranked[False]) & set(ranked[True])))
        rows.append({'method': method, 'label': METHODS[method],
                     'mean_top10_length_no_l2': float(np.mean(lengths[False])) if lengths[False] else 0.0,
                     'mean_top10_length_l2': float(np.mean(lengths[True])) if lengths[True] else 0.0,
                     'mean_top10_overlap': float(np.mean(overlaps)) if overlaps else 0.0})
    return rows


# ---------------------------------------------------------------------------
# 比較模式：同一查詢，不同 weighting 的 Top 10 並排，並標示相對於基準方法的名次變化
# ---------------------------------------------------------------------------

def rank_comparison(index, query, use_l2=True, baseline='logtfidf', methods=None):
    methods = list(methods or METHODS)
    documents = index.documents()
    rankings = {method: index.search(query, method, use_l2, limit=10)['ranked'] for method in methods}
    base_rows = [row for row, _ in rankings.get(baseline, [])]
    columns = []
    for method in methods:
        items = []
        for rank, (row, score) in enumerate(rankings[method], 1):
            base_rank = base_rows.index(row) + 1 if row in base_rows else None
            # change > 0：比基準方法名次往前（變好）；None：不在基準方法的 Top 10
            items.append({'rank': rank, 'pmid': index.pmids[row], 'doc_id': index.doc_ids[row],
                          'title': documents[row]['title'], 'score': score,
                          'change': None if base_rank is None else base_rank - rank})
        overlap = len(set(base_rows) & {row for row, _ in rankings[method]})
        columns.append({'method': method, 'label': METHODS[method], 'items': items,
                        'is_baseline': method == baseline, 'overlap': overlap})
    return {'query': query, 'use_l2': use_l2, 'baseline': baseline, 'columns': columns}


# ---------------------------------------------------------------------------
# 輸出：CSV 與 PNG
# ---------------------------------------------------------------------------

def _write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (round(value, 6) if isinstance(value, float) else value) for key, value in row.items()})


def _style_axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# 取出某個組合的平均指標值
def _mean_value(means, method, use_l2, key):
    return next(m[key] for m in means if m['method'] == method and m['use_l2'] == use_l2)


# 指標總覽熱圖：列 = 12 個組合，欄 = 5 個指標；每格印數值，顏色越深分數越高（每欄各自比較），每欄最高分加框
def plot_metrics(means, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rows = [(method, use_l2) for method in METHODS for use_l2 in (False, True)]
    labels = [f"{METHODS[m]}  {'+ L2' if l2 else '(no L2)'}" for m, l2 in rows]
    keys = list(METRICS)
    values = np.array([[_mean_value(means, m, l2, key) for key in keys] for m, l2 in rows])
    # 每一欄各自縮放到 0–1，顏色才看得出同一指標內的差異
    low, high = values.min(axis=0), values.max(axis=0)
    scaled = (values - low) / np.where(high > low, high - low, 1)
    cmap = LinearSegmentedColormap.from_list('blues', ['#f2f7fb', COLOR_L2])

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.imshow(scaled, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            best = abs(values[i, j] - high[j]) < 1e-9
            ax.text(j, i, f'{values[i, j]:.3f}', ha='center', va='center', fontsize=12,
                    color='white' if scaled[i, j] > 0.6 else INK, fontweight='bold' if best else 'normal')
            if best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
    ax.set_xticks(range(len(keys)), list(METRICS.values()), fontsize=12, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), labels, fontsize=11.5, color=INK)
    # 每個方法之間畫一條白色分隔線，讓「無 L2 / L2」兩列成對
    for boundary in range(2, len(rows), 2):
        ax.axhline(boundary - 0.5, color='white', linewidth=4)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f'Mean retrieval metrics · {len(EVALUATION_QUERIES)} queries · MeSH relevance\n'
                 'darker = higher within each column, boxed = best', loc='left', fontsize=12.5, color=INK, pad=34)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 在一個子圖上畫「無 L2 → 有 L2」的啞鈴圖：橘點 = 無 L2，藍點 = 有 L2，箭頭方向 = 正規化後的變化
def _draw_l2_dumbbell(ax, means, key, fontsize=11):
    methods = list(METHODS)
    before = np.array([_mean_value(means, m, False, key) for m in methods])
    after = np.array([_mean_value(means, m, True, key) for m in methods])
    positions = np.arange(len(methods))[::-1]
    span = max(before.max(), after.max()) - min(before.min(), after.min())
    pad = max(span * 0.12, 0.01)
    for y, a, b in zip(positions, before, after):
        if abs(b - a) > span * 0.04:  # 兩點太近時不畫箭頭，避免殘影
            ax.annotate('', xy=(b, y), xytext=(a, y),
                        arrowprops=dict(arrowstyle='-|>', color='#9aa9b4', lw=1.6, shrinkA=6, shrinkB=6))
    ax.scatter(before, positions, s=90, color=COLOR_NO_L2, zorder=3, label='no L2 (dot product)', edgecolor='white', linewidth=1.5)
    ax.scatter(after, positions, s=90, color=COLOR_L2, zorder=3, label='L2 (cosine)', edgecolor='white', linewidth=1.5)
    for y, a, b in zip(positions, before, after):
        # 數值標在點的外側：較小的值放左邊、較大的放右邊，避免和箭頭重疊
        (left, left_color), (right, right_color) = sorted([(a, COLOR_NO_L2), (b, COLOR_L2)], key=lambda item: item[0])
        ax.text(left - pad * 0.35, y, f'{left:.3f}', ha='right', va='center', fontsize=fontsize - 1, color=left_color)
        ax.text(right + pad * 0.35, y, f'{right:.3f}', ha='left', va='center', fontsize=fontsize - 1, color=right_color)
        # 最右邊一欄：L2 後的變化量（綠 = 變好，紅 = 變差）
        delta = b - a
        ax.text(1.0, y, f'{delta:+.3f}', transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=fontsize - 1, fontweight='bold',
                color='#0b7a4b' if delta >= 0.0005 else ('#b0412e' if delta <= -0.0005 else MUTED))
    ax.set_yticks(positions, [METHODS[m] for m in methods], fontsize=fontsize, color=INK)
    ax.set_xlim(min(before.min(), after.min()) - pad * 2.2, max(before.max(), after.max()) + pad * 4.5)
    ax.set_ylim(-0.7, len(methods) - 0.3)
    _style_axes(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)


# 單一指標的 L2 前後比較圖（例如 MRR）
def plot_l2_effect(means, key, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    _draw_l2_dumbbell(ax, means, key, fontsize=12)
    ax.set_xlabel(f'Mean {METRICS[key]} over {len(EVALUATION_QUERIES)} queries', color=MUTED, fontsize=11)
    ax.text(1.0, 1.01, 'change', transform=ax.transAxes, ha='right', va='bottom', fontsize=10.5, color=MUTED)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=11)
    ax.set_title(f'{METRICS[key]} before and after L2 normalization', loc='left', fontsize=14, color=INK, pad=34)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 五個指標的 L2 前後比較（small multiples：每個指標一張小圖，各自的 x 軸範圍）
def plot_l2_effect_all(means, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 3.1 * len(METRICS)))
    for ax, key in zip(axes, METRICS):
        _draw_l2_dumbbell(ax, means, key, fontsize=11)
        ax.set_title(METRICS[key], loc='left', fontsize=13, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, frameon=False, fontsize=11)
    fig.suptitle('Effect of L2 normalization on each metric (arrow: no L2 → L2)', x=0.02, ha='left', fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 文件長度分布直方圖
def plot_length_histogram(index, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(index.doc_lengths, bins=25, color=COLOR_L2, edgecolor='white', linewidth=1)
    ax.axvline(index.avg_length, color=INK, linewidth=1.2, linestyle='--')
    ax.text(index.avg_length, ax.get_ylim()[1] * 0.95, f'  mean = {index.avg_length:.1f}', color=INK, fontsize=9, va='top')
    ax.set_xlabel('Document length (tokens after preprocessing)', color=MUTED, fontsize=9)
    ax.set_ylabel('Documents', color=MUTED, fontsize=9)
    ax.set_title(f'Abstract length distribution (N = {index.n_documents})', loc='left', fontsize=11, color=INK)
    _style_axes(ax)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# RQ1 圖：tf 增加時 raw TF 線性成長、log TF 緩慢成長
def plot_tf_scaling(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tf = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(tf, tf, color=COLOR_NO_L2, linewidth=2, marker='o', markersize=4)
    ax.plot(tf, 1 + np.log10(tf), color=COLOR_L2, linewidth=2, marker='o', markersize=4)
    ax.text(20.3, 20, 'Raw TF = tf', color=INK, fontsize=9, va='center')
    ax.text(20.3, 1 + np.log10(20), '1 + log10(tf)', color=INK, fontsize=9, va='center')
    ax.set_xlim(0, 25)
    ax.set_xlabel('tf (term occurrences in one document)', color=MUTED, fontsize=9)
    ax.set_ylabel('Term weight', color=MUTED, fontsize=9)
    ax.set_title('RQ1 · Raw TF vs Log TF', loc='left', fontsize=11, color=INK)
    _style_axes(ax)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 一次產生所有實驗結果
def run_all(index, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = index.summary()
    evaluation = evaluate_all(index)

    # 任務一：詞項統計
    _write_csv(output / 'corpus_summary.csv', [{'preprocessing': index.config.label, **summary}])
    _write_csv(output / 'top_df_terms.csv', index.top_df_terms(50))
    _write_csv(output / 'doc_length_distribution.csv', index.length_histogram(20))
    plot_length_histogram(index, output / 'doc_length_distribution.png')

    # 任務四：RQ1 / RQ2 / RQ4 示範
    _write_csv(output / 'rq1_tf_scaling.csv', tf_scaling_table())
    plot_tf_scaling(output / 'rq1_tf_vs_logtf.png')
    _write_csv(output / 'rq1_rq2_term_weights.csv', term_weight_demo(index))
    idf_rows = idf_high_low(index)
    _write_csv(output / 'rq2_idf_high_vs_low_df.csv',
               [{'group': 'high_df', **row} for row in idf_rows['high']] +
               [{'group': 'low_df', **row} for row in idf_rows['low']])
    length = length_demo(index)
    _write_csv(output / 'rq4_length_demo.csv',
               [{'setting': 'no_l2', **{k: v for k, v in row.items() if k != 'row'}} for row in length['no_l2']] +
               [{'setting': 'l2', **{k: v for k, v in row.items() if k != 'row'}} for row in length['l2']])
    length_bias = length_bias_table(index)
    _write_csv(output / 'rq4_top10_length_by_method.csv', length_bias)

    # 任務五：評估
    _write_csv(output / 'relevance_sets.csv', evaluation['queries'])
    _write_csv(output / 'per_query_metrics.csv', evaluation['per_query'])
    _write_csv(output / 'mean_metrics.csv', evaluation['means'])
    plot_metrics(evaluation['means'], output / 'metrics_comparison.png')
    plot_l2_effect(evaluation['means'], 'mrr', output / 'l2_effect_mrr.png')
    plot_l2_effect_all(evaluation['means'], output / 'l2_effect_all_metrics.png')

    # 每個查詢 × 每個組合的 Top 10 排名（附相關等級，方便人工檢查）
    ranking_rows = []
    for query in EVALUATION_QUERIES:
        grades = relevance_grades(index, query)
        for method, use_l2 in configurations():
            for rank, row in enumerate(index.ranking(query['text'], method, use_l2, limit=10), 1):
                ranking_rows.append({'query_id': query['id'], 'config': config_label(method, use_l2),
                                     'rank': rank, 'pmid': index.pmids[row], 'relevance': int(grades[row]),
                                     'doc_length': int(index.doc_lengths[row])})
    _write_csv(output / 'top10_rankings.csv', ranking_rows)

    result = {'preprocessing': index.config.label, 'summary': summary, 'means': evaluation['means'],
              'queries': evaluation['queries'], 'length_bias': length_bias,
              'length_demo': {k: length[k] for k in ('query', 'method', 'mean_length_no_l2', 'mean_length_l2',
                                                      'corpus_mean_length', 'overlap')},
              'required_methods': list(REQUIRED_METHODS), 'formulas': schemes.FORMULAS}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
