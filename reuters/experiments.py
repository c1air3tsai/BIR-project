# =============================================================================
# reuters/experiments.py
# Project 4：產生所有實驗結果（CSV + PNG），輸出到 docs/project4_results/
#
# 三種切分（ModApte / ModLewis / ModHayes）各跑一次，結果放在 docs/project4_results/<切分>/；
# 三種切分共用的檔案放在 docs/project4_results/ 最上層：
#     toy_word2vec_loss.png    numpy 自製 CBOW / Skip-Gram 的 loss 曲線（與切分無關）
#     split_comparison.csv/.png  三種切分的檢索結果比較
#     summary.json             所有切分的重點數字彙整（報告引用）
#
# 每個切分資料夾裡：
#
#   Word-level analysis（詞層級）：
#     neighbors.csv            每個詞的 Top-5 鄰近詞（CBOW / Skip-Gram）
#     embedding_pca.png        PCA 2D 投影（保留全域變異）
#     embedding_tsne.png       t-SNE 2D 投影（保留局部鄰近結構）
#     projection.json          上面兩種投影的座標（網頁互動圖用，可縮放）
#     w2v_training.csv/.png    gensim 每個 epoch 的訓練 loss
#     topic_keywords.csv       每個主題在訓練集裡最有代表性的詞（keyword 查詢的依據）
#   Document-level retrieval（文件層級）：
#     mean_metrics.csv、metrics_by_type.csv、metrics_by_topic.csv、per_query_metrics.csv
#     metrics_comparison.png   方法 × 指標熱圖
#     keyword_vs_paraphrase.png  兩種查詢類型的 nDCG@10
#     query_comparison.csv     每個查詢 A vs B 的差異（找出 B 贏 / A 贏的例子）
#     corpus_size.csv          Word2Vec 訓練語料加大（train → train + unused）對檢索的影響
#     summary.json             這個切分的重點數字
# =============================================================================
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from weighting.experiments import GRID, INK, MUTED, _style_axes

from . import embedding, toy_word2vec
from .corpus import SPLITS, TOPICS, document_text, split_counts, split_documents
from .evaluation import METRICS, evaluate_all
from .queries import EVALUATION_QUERIES, NEIGHBOR_WORDS
from .retrieval import CORE_METHODS, METHODS, load_index
from .text import preprocess

# Okabe-Ito 色盲友善配色（與網站其他圖表一致）
METHOD_COLORS = {'tfidf': '#D55E00', 'w2v_tfidf_cbow': '#56B4E9', 'w2v_tfidf_sg': '#0072B2',
                 'w2v_mean_cbow': '#009E73', 'w2v_mean_sg': '#007A5A',
                 'hybrid_cbow': '#CC79A7', 'hybrid_sg': '#A0527F'}
ARCH_COLORS = {'cbow': '#56B4E9', 'skipgram': '#0072B2'}
SPLIT_COLORS = {'modapte': '#0072B2', 'modlewis': '#E69F00', 'modhayes': '#009E73'}
TOPIC_COLORS = {'earn': '#E69F00', 'acq': '#CC79A7', 'crude': '#D55E00', 'grain': '#009E73', 'trade': '#0072B2'}
# 鄰近詞分析的每個詞屬於哪個主題（圖上著色用）
WORD_TOPIC = {'oil': 'crude', 'petroleum': 'crude', 'opec': 'crude', 'wheat': 'grain', 'corn': 'grain',
              'merger': 'acq', 'takeover': 'acq', 'profit': 'earn', 'loss': 'earn', 'dividend': 'earn',
              'tariff': 'trade', 'deficit': 'trade', 'japan': 'trade'}


def _write_csv(path, rows):
    rows = list(rows)
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else [], extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Word-level analysis
# ---------------------------------------------------------------------------

def neighbor_rows(split, words=NEIGHBOR_WORDS, topn=5):
    rows = []
    for word in words:
        for architecture in embedding.ARCHITECTURES:
            for rank, item in enumerate(embedding.neighbors(word, split, architecture, topn) or [], 1):
                rows.append({'word': word, 'architecture': architecture, 'rank': rank,
                             'neighbor': item['word'], 'cosine': item['cosine']})
    return rows


# 把種子詞與它們的 Top-5 鄰近詞投影到 2D；method = 'pca' 或 'tsne'
# 回傳 [{'word', 'seed'（屬於哪個種子詞）, 'topic', 'is_seed', 'cosine'（與種子詞）, 'x', 'y'}, ...]
def projection_points(method, architecture, split, words=NEIGHBOR_WORDS):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    vectors, _ = embedding.load(split, 'train', architecture)
    points = {}
    for word in words:
        if word not in vectors.key_to_index:
            continue
        points.setdefault(word, {'word': word, 'seed': word, 'is_seed': True, 'cosine': 1.0})
        for item in embedding.neighbors(word, split, architecture, 5) or []:
            points.setdefault(item['word'], {'word': item['word'], 'seed': word, 'is_seed': False,
                                             'cosine': item['cosine']})
    rows = list(points.values())
    matrix = np.vstack([vectors[row['word']] for row in rows])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    if method == 'pca':
        xy = PCA(n_components=2, random_state=42).fit_transform(matrix)
    else:
        xy = TSNE(n_components=2, perplexity=min(15, len(rows) - 1), random_state=42,
                  init='pca', metric='cosine').fit_transform(matrix)
    for row, (x, y) in zip(rows, xy):
        row.update({'topic': WORD_TOPIC[row['seed']], 'x': round(float(x), 4), 'y': round(float(y), 4)})
    return rows


# 網頁互動圖用：兩種投影 × 兩種架構的座標，存成 projection.json
def projection_data(split):
    return {method: {architecture: projection_points(method, architecture, split)
                     for architecture in embedding.ARCHITECTURES} for method in ('tsne', 'pca')}


def plot_projection(method, path, data):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, architecture in zip(axes, embedding.ARCHITECTURES):
        for row in data[method][architecture]:
            seed = row['is_seed']
            ax.scatter(row['x'], row['y'], s=70 if seed else 22, color=TOPIC_COLORS[row['topic']],
                       edgecolor='white', linewidth=0.8, zorder=3)
            ax.annotate(row['word'], (row['x'], row['y']), xytext=(3, 3), textcoords='offset points',
                        fontsize=9.5 if seed else 7.5, fontweight='bold' if seed else 'normal',
                        color=INK if seed else MUTED)
        ax.set_title(embedding.ARCHITECTURES[architecture], loc='left', fontsize=13, color=INK)
        _style_axes(ax)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=f'{t} seed words')
               for t, c in TOPIC_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=10)
    title = 'PCA (global variance)' if method == 'pca' else 't-SNE (local neighborhoods)'
    fig.suptitle(f'Word2Vec {title}: seed words (bold) and their Top-5 neighbors', x=0.02, ha='left',
                 fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_loss(metas, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (architecture, meta) in zip(axes, metas.items()):
        losses = meta['epoch_losses']
        ax.plot(range(1, len(losses) + 1), losses, color=ARCH_COLORS[architecture], linewidth=2, marker='o', markersize=3)
        ax.set_title(f"{embedding.ARCHITECTURES[architecture]} · {meta['seconds']} s", loc='left', fontsize=12, color=INK)
        ax.set_xlabel('epoch', color=MUTED, fontsize=9)
        ax.set_ylabel('training loss (gensim)', color=MUTED, fontsize=9)
        _style_axes(ax)
        ax.grid(axis='y', color=GRID)
    fig.suptitle('Word2Vec training loss per epoch (scales differ between architectures)', x=0.02, ha='left',
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_toy_loss(result, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for architecture in ('cbow', 'skipgram'):
        losses = result[architecture]['losses']
        ax.plot(range(1, len(losses) + 1), losses, color=ARCH_COLORS[architecture], linewidth=2,
                label=embedding.ARCHITECTURES[architecture])
        ax.text(len(losses) + 3, losses[-1], f'{embedding.ARCHITECTURES[architecture]} {losses[-1]:.2f}',
                va='center', fontsize=9, color=INK)
    ax.set_xlim(0, len(result['cbow']['losses']) * 1.25)
    ax.set_xlabel('epoch', color=MUTED, fontsize=9)
    ax.set_ylabel('cross-entropy loss', color=MUTED, fontsize=9)
    ax.set_title('Our numpy CBOW / Skip-Gram on a toy corpus (full softmax)', loc='left', fontsize=12, color=INK)
    _style_axes(ax)
    ax.grid(axis='y', color=GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Document-level retrieval
# ---------------------------------------------------------------------------

def plot_metric_heatmap(means, path):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    keys = list(METRICS)
    values = np.array([[row[k] for k in keys] for row in means])
    low, high = values.min(axis=0), values.max(axis=0)
    scaled = (values - low) / np.where(high > low, high - low, 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.imshow(scaled, cmap=LinearSegmentedColormap.from_list('b', ['#f2f7fb', '#0072B2']), aspect='auto', vmin=0, vmax=1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            best = abs(values[i, j] - high[j]) < 1e-9
            ax.text(j, i, f'{values[i, j]:.3f}', ha='center', va='center', fontsize=12,
                    color='white' if scaled[i, j] > 0.6 else INK, fontweight='bold' if best else 'normal')
            if best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
    ax.set_xticks(range(len(keys)), list(METRICS.values()), fontsize=12, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(means)), [row['label'] for row in means], fontsize=11.5, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f'Mean retrieval metrics · {len(EVALUATION_QUERIES)} queries · relevance = Reuters topic\n'
                 'darker = higher within each column, boxed = best', loc='left', fontsize=12.5, color=INK, pad=34)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 每個方法在 keyword / paraphrase 兩種查詢的 nDCG@10（啞鈴圖）
def plot_query_types(by_type, path):
    plt = _plt()
    methods = list(METHODS)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(methods))[::-1]
    for y, method in zip(positions, methods):
        k = next(r['ndcg10'] for r in by_type if r['method'] == method and r['type'] == 'keyword')
        p = next(r['ndcg10'] for r in by_type if r['method'] == method and r['type'] == 'paraphrase')
        ax.plot([p, k], [y, y], color='#c9d4dc', linewidth=2, zorder=1)
        ax.scatter([k], [y], s=90, color='#0072B2', zorder=3, edgecolor='white', label='keyword queries' if y == positions[0] else None)
        ax.scatter([p], [y], s=90, color='#D55E00', zorder=3, edgecolor='white', label='paraphrase queries' if y == positions[0] else None)
        ax.text(k + 0.012, y, f'{k:.2f}', va='center', fontsize=10, color='#0072B2')
        ax.text(p - 0.012, y, f'{p:.2f}', va='center', ha='right', fontsize=10, color='#D55E00')
    ax.set_yticks(positions, [METHODS[m] for m in methods], fontsize=11, color=INK)
    ax.set_xlim(0.35, 1.05)
    ax.set_xlabel('mean nDCG@10', color=MUTED, fontsize=10)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10.5)
    ax.set_title('Keyword vs paraphrase queries', loc='left', fontsize=13, color=INK, pad=30)
    _style_axes(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 每個查詢：A（TF-IDF）vs B（TF-IDF weighted Skip-Gram / CBOW）的差異
def query_comparison(per_query):
    rows = []
    for query in EVALUATION_QUERIES:
        get = lambda m, k: next(r[k] for r in per_query if r['query_id'] == query['id'] and r['method'] == m)
        rows.append({'query_id': query['id'], 'topic': query['topic'], 'type': query['type'], 'text': query['text'],
                     'ndcg_tfidf': get('tfidf', 'ndcg10'), 'ndcg_cbow': get('w2v_tfidf_cbow', 'ndcg10'),
                     'ndcg_sg': get('w2v_tfidf_sg', 'ndcg10'),
                     'delta_sg_minus_tfidf': get('w2v_tfidf_sg', 'ndcg10') - get('tfidf', 'ndcg10'),
                     'p10_tfidf': get('tfidf', 'p10'), 'p10_sg': get('w2v_tfidf_sg', 'p10')})
    return rows


# 每個主題在訓練集裡最有代表性的詞（keyword 查詢就是用這類詞寫的）：
#   score = 含這個詞的文件比例（主題內）− 含這個詞的文件比例（主題外），越大越常見、越只出現在這個主題
def topic_keywords(split, n=10):
    docs = split_documents('train', split)
    term_sets = [set(preprocess(document_text(doc))) for doc in docs]
    rows = []
    for topic in TOPICS:
        inside = [terms for terms, doc in zip(term_sets, docs) if topic in doc['topics']]
        outside = [terms for terms, doc in zip(term_sets, docs) if topic not in doc['topics']]
        if not inside or not outside:
            continue
        df_in = Counter(term for terms in inside for term in terms)
        df_out = Counter(term for terms in outside for term in terms)
        scored = sorted(((df_in[t] / len(inside) - df_out[t] / len(outside), t) for t in df_in), key=lambda x: (-x[0], x[1]))
        for rank, (score, term) in enumerate(scored[:n], 1):
            rows.append({'topic': topic, 'rank': rank, 'term': term, 'in_topic': df_in[term] / len(inside),
                         'outside': df_out[term] / len(outside), 'score': score})
    return rows


# 三種切分的檢索結果比較（核心方法 A / B 的平均指標，與 keyword / paraphrase 的 nDCG@10）
def split_comparison(results):
    rows = []
    for split, result in results.items():
        for row in result['means']:
            types = {r['type']: r['ndcg10'] for r in result['by_type'] if r['method'] == row['method']}
            rows.append({'split': split, 'method': row['method'], 'label': row['label'],
                         **{k: row[k] for k in METRICS},
                         'ndcg_keyword': types.get('keyword', 0.0), 'ndcg_paraphrase': types.get('paraphrase', 0.0)})
    return rows


# 三種切分 × 核心方法的分組橫條圖：P@10、MRR、nDCG@10 各一張小圖（數值直接標在條的尾端）
def plot_split_comparison(rows, path):
    plt = _plt()
    keys = ('p10', 'mrr', 'ndcg10')
    splits = [s for s in SPLITS if any(r['split'] == s for r in rows)]
    fig, axes = plt.subplots(1, len(keys), figsize=(13, 4.6), sharey=True)
    bar, gap = 0.24, 0.02
    positions = np.arange(len(CORE_METHODS))[::-1]
    for ax, key in zip(axes, keys):
        for k, split in enumerate(splits):
            offset = (len(splits) - 1) / 2 * (bar + gap) - k * (bar + gap)
            for y, method in zip(positions, CORE_METHODS):
                value = next(r[key] for r in rows if r['split'] == split and r['method'] == method)
                ax.barh(y + offset, value, height=bar, color=SPLIT_COLORS[split], edgecolor='white', linewidth=0,
                        label=SPLITS[split]['label'] if y == positions[0] else None)
                ax.text(value + 0.012, y + offset, f'{value:.2f}', va='center', fontsize=9, color=INK)
        ax.set_title(METRICS[key], loc='left', fontsize=12.5, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        _style_axes(ax)
        ax.grid(axis='x', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(positions, [METHODS[m] for m in CORE_METHODS], fontsize=11, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=len(splits), frameon=False, fontsize=10.5)
    fig.suptitle(f'Same {len(EVALUATION_QUERIES)} queries on three official splits (relevance = Reuters topic)',
                 x=0.02, ha='left', fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 一個切分的完整實驗，輸出到 output（docs/project4_results/<切分>/）
def run_split(split, output, include_corpus_size=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = load_index(split, 'train')
    summary = index.summary()

    # 詞層級
    metas = {a: embedding.load(split, 'train', a)[1] for a in embedding.ARCHITECTURES}
    _write_csv(output / 'w2v_training.csv',
               [{'architecture': a, 'epoch': i, 'loss': loss} for a, m in metas.items()
                for i, loss in enumerate(m['epoch_losses'], 1)])
    plot_training_loss(metas, output / 'w2v_training.png')
    _write_csv(output / 'neighbors.csv', neighbor_rows(split))
    projection = projection_data(split)
    (output / 'projection.json').write_text(json.dumps(projection, ensure_ascii=False), encoding='utf-8')
    plot_projection('pca', output / 'embedding_pca.png', projection)
    plot_projection('tsne', output / 'embedding_tsne.png', projection)
    _write_csv(output / 'topic_keywords.csv', topic_keywords(split))

    # 文件層級
    evaluation = evaluate_all(index)
    _write_csv(output / 'per_query_metrics.csv', evaluation['per_query'])
    _write_csv(output / 'mean_metrics.csv', evaluation['means'])
    _write_csv(output / 'metrics_by_type.csv', evaluation['by_type'])
    _write_csv(output / 'metrics_by_topic.csv', evaluation['by_topic'])
    plot_metric_heatmap(evaluation['means'], output / 'metrics_comparison.png')
    plot_query_types(evaluation['by_type'], output / 'keyword_vs_paraphrase.png')
    _write_csv(output / 'query_comparison.csv', query_comparison(evaluation['per_query']))

    # 語料大小：Word2Vec 改用 train + unused 訓練，只重算方法 B（ModHayes 沒有未使用的文件，跳過）
    corpus_rows = []
    if include_corpus_size and split_counts()[split]['unused']:
        bigger = evaluate_all(load_index(split, 'train_unused'), methods=['w2v_tfidf_cbow', 'w2v_tfidf_sg'])
        for corpus, means in (('train', evaluation['means']), ('train_unused', bigger['means'])):
            for row in means:
                if row['method'] in ('w2v_tfidf_cbow', 'w2v_tfidf_sg'):
                    meta = embedding.load(split, corpus, 'skipgram' if row['method'].endswith('sg') else 'cbow')[1]
                    corpus_rows.append({'w2v_corpus': corpus, 'method': row['method'], 'tokens': meta['tokens'],
                                        'vocabulary': meta['vocabulary'], **{k: row[k] for k in METRICS}})
    _write_csv(output / 'corpus_size.csv', corpus_rows)

    result = {'split': split, 'counts': split_counts()[split], 'dataset': summary,
              'w2v': {a: {k: m[k] for k in ('sentences', 'tokens', 'vocabulary', 'seconds')} for a, m in metas.items()},
              'params': embedding.PARAMS, 'means': evaluation['means'], 'by_type': evaluation['by_type'],
              'corpus_size': corpus_rows, 'core_methods': list(CORE_METHODS)}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


# 全部重跑：splits 指定要跑哪些切分；沒跑到的切分讀之前的 summary.json，讓比較表保持完整
def run_all(output_dir, include_corpus_size=True, splits=tuple(SPLITS)):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_toy_loss(toy_word2vec.demo(), output / 'toy_word2vec_loss.png')
    results = {}
    for split in SPLITS:
        if split in splits:
            results[split] = run_split(split, output / split, include_corpus_size)
        elif (output / split / 'summary.json').is_file():
            results[split] = json.loads((output / split / 'summary.json').read_text(encoding='utf-8'))
    comparison = split_comparison(results)
    _write_csv(output / 'split_comparison.csv', comparison)
    plot_split_comparison(comparison, output / 'split_comparison.png')
    (output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    return results
