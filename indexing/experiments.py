# =============================================================================
# indexing/experiments.py
# Project 5：BSBI 與 SPIMI 的效能比較與檢索測試，輸出到 docs/project5_results/
#
# 公平比較：兩種方法用同一批文件（collection.read_documents）、同一套前處理、同一個記憶體上限、
#           同樣的磁碟格式與合併程式（storage.py），只差在建索引的方式。
# Baseline（baseline.py）：同樣的文件與前處理，但不限制記憶體、整份索引放在記憶體裡建好再寫出，
#           當作「沒有記憶體限制時」的參考；改變記憶體上限的實驗不跑它（它沒有上限可以改）。
#
# 量測方式：
#   建構時間   time.perf_counter；同一設定跑 repeats 次取中位數，並拆成 5 個階段
#   記憶體     tracemalloc 的峰值（Python 在建索引期間配置的記憶體）；tracemalloc 會拖慢速度，
#              所以和計時分開跑（同一設定再跑一次）
#   blocks / 中間檔 / 最終索引大小：直接數 block 檔與量檔案大小
#
# 輸出：
#   summary.json            全部數字與中間結果（網頁讀這個）
#   comparison.csv          完整語料、預設記憶體上限下的比較表
#   memory_sweep.csv/.png   改變記憶體上限（1–64 MB）
#   scaling.csv/.png        改變文件數（2,500 → 1,000,000 篇）
#   stage_times.png         三種方法各階段的時間
#   retrieval.csv           5 組查詢 + 作業的 5 組範例查詢在 BSBI、SPIMI（與 baseline）索引的結果是否一致
# 最終索引留在 data/indexing/full/<方法>/index/（網頁的即時查詢用）
# =============================================================================
import csv
import gc
import json
import shutil
import statistics
import tempfile
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
from django.conf import settings

from . import baseline, bsbi, spimi
from .collection import corpus_size, read_documents
from .inverted_index import DiskIndex, InvertedIndex
from .queries import EXAMPLE_QUERIES, QUERIES, TRACE_TERMS
from .sources import DEFAULT_SOURCE, get as get_source
from .storage import index_size


# 每個資料來源（sources.py）的實驗結果與最終索引放在不同資料夾
def results_dir(source=DEFAULT_SOURCE):
    return get_source(source)['results_dir']


def work_dir(source=DEFAULT_SOURCE):
    return get_source(source)['work_dir']


RESULTS_DIR = results_dir('reuters')
# 網頁要用的最終索引留在 data/indexing/（PubMed 在 data/indexing/pubmed/）的 full/；計時用的暫存索引放系統暫存資料夾
# （Docker 裡是容器自己的磁碟，避免 Windows 掛載資料夾的檔案分享拖慢寫檔，影響計時）
WORK_DIR = work_dir('reuters')
TMP_DIR = Path(tempfile.gettempdir()) / 'bir_indexing'
MB = 1 << 20
DEFAULT_LIMIT_MB = 4
SWEEP_MB = (1, 2, 4, 8, 16, 32, 64)
# 規模實驗的文件數：語料篇數（Reuters 21,578、PubMed 20,000）以下是前 N 篇（真實資料），以上是重複讀語料
SCALES = (2500, 5000, 10000, 21578, 100000, 1000000)


def scales_for(source=DEFAULT_SOURCE):
    size = corpus_size(source)
    return tuple(sorted({n for n in SCALES if n < size or n > 21578} | {size}))
# tracemalloc 會讓建索引慢約 5 倍，超過這個文件數就只計時、不量記憶體峰值（100 萬篇量一次要超過 1 小時）
TRACE_MAX_DOCS = 100000
METHODS = {'baseline': baseline, 'bsbi': bsbi, 'spimi': spimi}
# 有記憶體上限、要切 blocks 的兩種方法（作業比較的主角）
LIMITED = ('bsbi', 'spimi')
LABELS = {'baseline': 'Baseline', 'bsbi': 'BSBI', 'spimi': 'SPIMI'}
COLORS = {'baseline': '#7f8c96', 'bsbi': '#D55E00', 'spimi': '#0072B2'}
STAGES = {'preprocess': '前處理', 'pairs': '產生 pairs／加進 dictionary', 'sort': '排序',
          'write': '寫 partial index（baseline：寫最終索引）', 'merge': '合併'}
STAGES_EN = {'preprocess': 'preprocessing', 'pairs': 'pairs (BSBI) / dictionary (SPIMI, baseline)', 'sort': 'sorting',
             'write': 'writing blocks (baseline: final index)', 'merge': 'merging'}


def _log(message):
    print(message, flush=True)


# 跑一次建索引，回傳時間、blocks、中間檔與最終索引的大小；trace = True 時改量 tracemalloc 峰值
def run_once(method, count, limit, folder, samples=False, trace=False, source=DEFAULT_SOURCE):
    corpus_size(source)                             # 先把語料讀進記憶體，不算在量測裡
    gc.collect()
    if trace:
        tracemalloc.start()
    result = METHODS[method].build(read_documents(count, source), folder, limit, samples=samples,
                                   trace_terms=TRACE_TERMS[source])
    if trace:
        result['peak'] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    blocks = result['blocks']
    sizes = index_size(Path(folder) / 'index')
    result.update({'limit': limit, 'block_count': len(blocks), 'intermediate_files': len(blocks),
                   'intermediate_bytes': sum(b['bytes'] for b in blocks), 'index_bytes': sizes,
                   'final_bytes': sizes['dictionary'] + sizes['postings'],
                   'block_entries': sum(b['pairs'] for b in blocks)})
    return result


def _clean(folder):
    shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# 完整語料、預設記憶體上限：比較表 + 中間結果
# ---------------------------------------------------------------------------

def full_comparison(limit, repeats, source=DEFAULT_SOURCE):
    rows = {}
    tmp = TMP_DIR / source
    for method in METHODS:
        folder = work_dir(source) / 'full' / method
        kept = run_once(method, None, limit, folder, samples=True, source=source)
        runs = [run_once(method, None, limit, tmp / method, source=source) for _ in range(repeats)]
        _clean(tmp / method)
        median = sorted(runs, key=lambda r: r['seconds'])[len(runs) // 2]
        peak = run_once(method, None, limit, tmp / method, trace=True, source=source)['peak']
        _clean(tmp / method)
        rows[method] = {**kept, 'seconds': median['seconds'], 'times': median['times'], 'peak': peak,
                        'runs': [r['seconds'] for r in runs], 'stdev': statistics.pstdev(r['seconds'] for r in runs)}
        _log(f"  {LABELS[method]:8s} {median['seconds']:.2f} s (runs {', '.join(f'{r:.2f}' for r in rows[method]['runs'])}) "
             f"blocks {kept['block_count']} peak {peak / MB:.1f} MB")
    return rows


# 三個最終索引（baseline、BSBI、SPIMI）與 Part 1 的記憶體索引逐一比較：詞彙相同、每個 term 的 postings 完全相同
def equivalence(reference, source=DEFAULT_SOURCE):
    disk = {method: DiskIndex(work_dir(source) / 'full' / method / 'index') for method in METHODS}
    vocabulary = reference.vocabulary()
    report = {'vocabulary': len(vocabulary), 'postings': reference.pairs()}
    for method, index in disk.items():
        report[method] = {'same_vocabulary': index.vocabulary() == vocabulary,
                          'different_postings': sum(list(index.postings(t)) != reference.postings(t) for t in vocabulary)}
    return report


def retrieval_rows(source=DEFAULT_SOURCE):
    disk = {method: DiskIndex(work_dir(source) / 'full' / method / 'index') for method in METHODS}
    rows = []
    for name, a, b in QUERIES[source] + EXAMPLE_QUERIES:
        results = {method: index.boolean(a, b) for method, index in disk.items()}
        for kind, label in (('a', a), ('b', b), ('and', f'{a} AND {b}'), ('or', f'{a} OR {b}')):
            first, second = results['bsbi'][kind], results['spimi'][kind]
            rows.append({'query': name, 'type': {'a': 'single', 'b': 'single', 'and': 'AND', 'or': 'OR'}[kind],
                         'expression': label, 'bsbi_hits': len(first), 'spimi_hits': len(second),
                         'baseline_hits': len(results['baseline'][kind]), 'identical': first == second,
                         'baseline_identical': list(results['baseline'][kind]) == list(first),
                         'first_docs': ' '.join(map(str, first[:10]))})
    return rows


# ---------------------------------------------------------------------------
# 改變記憶體上限、改變文件數
# ---------------------------------------------------------------------------

def _measure(method, count, limit, label, source=DEFAULT_SOURCE):
    folder = TMP_DIR / source / method
    timed = run_once(method, count, limit, folder, source=source)
    _clean(folder)
    peak = None
    if timed['documents'] <= TRACE_MAX_DOCS:
        peak = run_once(method, count, limit, folder, trace=True, source=source)['peak']
        _clean(folder)
    times = timed['times']
    row = {'method': method, 'documents': timed['documents'], 'tokens': timed['tokens'], 'vocabulary': timed['vocabulary'],
           'limit_mb': limit / MB, 'seconds': timed['seconds'], 'index_seconds': timed['seconds'] - times['preprocess'],
           **{f'{stage}_seconds': value for stage, value in times.items()},
           'blocks': timed['block_count'], 'intermediate_files': timed['intermediate_files'],
           'intermediate_bytes': timed['intermediate_bytes'], 'final_bytes': timed['final_bytes'],
           'block_entries': timed['block_entries'], 'peak_bytes': peak, 'term_id_map_bytes': timed['term_id_map_bytes'],
           'index_memory': timed.get('index_memory')}
    peak_text = f'{peak / MB:6.1f} MB' if peak is not None else '     —   '
    _log(f"  {label:>12s} {LABELS[method]:8s} {row['seconds']:7.2f} s  blocks {row['blocks']:5d}  "
         f"peak {peak_text}  intermediate {row['intermediate_bytes'] / MB:7.1f} MB")
    return row


def memory_sweep(limits_mb, source=DEFAULT_SOURCE):
    return [_measure(method, None, mb * MB, f'{mb} MB', source) for mb in limits_mb for method in LIMITED]


def scaling(counts, limit, source=DEFAULT_SOURCE):
    return [_measure(method, count, limit, f'{count:,} docs', source) for count in counts for method in METHODS]


# Heaps' law：V = K · T^β，用真實的語料子集（不含重複讀的部分）擬合，推估更大的語料有多少詞
def heaps_fit(rows, source=DEFAULT_SOURCE):
    real = [r for r in rows if r['method'] == 'bsbi' and r['documents'] <= corpus_size(source)]
    if len(real) < 2:
        return None
    beta, log_k = np.polyfit(np.log([r['tokens'] for r in real]), np.log([r['vocabulary'] for r in real]), 1)
    per_doc = real[-1]['tokens'] / real[-1]['documents']
    bytes_per_term = real[-1]['term_id_map_bytes'] / real[-1]['vocabulary']
    estimates = []
    for documents in (100000, 1000000):
        vocabulary = float(np.exp(log_k) * (per_doc * documents) ** beta)
        estimates.append({'documents': documents, 'tokens': per_doc * documents, 'vocabulary': vocabulary,
                          'term_id_map_bytes': vocabulary * bytes_per_term})
    return {'k': float(np.exp(log_k)), 'beta': float(beta), 'estimates': estimates}


# ---------------------------------------------------------------------------
# 實作複雜度：兩個檔案的程式行數（不含空行與註解）與流程的差異
# ---------------------------------------------------------------------------

def code_lines(module):
    lines = Path(module.__file__).read_text(encoding='utf-8').splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))


def complexity():
    from . import storage
    return {'baseline': code_lines(baseline), 'bsbi': code_lines(bsbi), 'spimi': code_lines(spimi),
            'shared': code_lines(storage)}


# ---------------------------------------------------------------------------
# 圖
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _axes_style(ax):
    from weighting.experiments import GRID, MUTED, _style_axes
    _style_axes(ax)
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED)


def plot_stages(rows, path, label='Reuters-21578'):
    from weighting.experiments import INK
    plt = _plt()
    order = [m for m in ('spimi', 'bsbi', 'baseline') if m in rows]          # 由下往上畫
    fig, ax = plt.subplots(figsize=(9.5, 1.2 + 0.9 * len(order)))
    shades = ['#cfd8de', '#9ec3dc', '#E69F00', '#56B4E9', '#009E73']
    for y, method in enumerate(order):
        left = 0
        for (stage, label), color in zip(STAGES_EN.items(), shades):
            value = rows[method]['times'][stage]
            ax.barh(y, value, left=left, color=color, edgecolor='white', label=label if y == 0 else None)
            if value > 0.25:
                ax.text(left + value / 2, y, f'{value:.2f}', ha='center', va='center', fontsize=9, color=INK)
            left += value
        ax.text(left + 0.06, y, f"{rows[method]['seconds']:.2f} s", va='center', fontsize=10.5, color=INK)
    ax.set_yticks(range(len(order)), [LABELS[m] if m != 'baseline' else 'Baseline\n(no limit)' for m in order], fontsize=11)
    ax.set_xlabel('seconds (median run)')
    ax.set_xlim(0, max(r['seconds'] for r in rows.values()) * 1.15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45 + 0.07 * len(order)), ncol=3, frameon=False, fontsize=8.5)
    _axes_style(ax)
    ax.set_title(f'Index construction time by stage (full {label})', loc='left', fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 四張小圖（時間、blocks、記憶體峰值、中間檔）；x 軸用對數刻度，但刻度直接標實際的值（ticks）
# log_y：文件數從 2,500 到 1,000,000，時間、blocks、中間檔也跨好幾個數量級，y 軸跟著用對數
#        （baseline 沒有 blocks 與中間檔，值是 0，對數軸上不畫）
# reference：{欄位: 值}，畫成灰色虛線（改變記憶體上限時，baseline 不受上限影響，當參考線）
def _line_panels(rows, x_key, x_label, path, title, ticks, log_y=False, rotate=0, reference=None):
    from matplotlib.ticker import FixedLocator, NullLocator
    from weighting.experiments import INK
    plt = _plt()
    panels = (('seconds', 'construction time (s)', 1), ('blocks', 'number of blocks', 1),
              ('peak_bytes', 'peak memory (MB, tracemalloc)', MB), ('intermediate_bytes', 'intermediate files (MB)', MB))
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.8))
    for ax, (key, label, scale) in zip(axes, panels):
        for method in METHODS:
            points = [r for r in rows if r['method'] == method and r[key] is not None
                      and not (log_y and key != 'peak_bytes' and r[key] <= 0)]
            if points:
                ax.plot([r[x_key] for r in points], [r[key] / scale for r in points], marker='o', linewidth=2,
                        color=COLORS[method], label=LABELS[method])
        if reference and reference.get(key):
            ax.axhline(reference[key] / scale, color=COLORS['baseline'], linestyle='--', linewidth=1.5,
                       label='Baseline (no limit)')
        ax.set_title(label, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator([value for value, _ in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([text for _, text in ticks], fontsize=8, rotation=rotate, ha='right' if rotate else 'center')
        if log_y and key != 'peak_bytes':
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        _axes_style(ax)
    axes[0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha='left', fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _short(count):
    return f'{count // 1000000}M' if count >= 1000000 else f'{count / 1000:.3g}k'


# 畫所有圖（run_all 跑完會呼叫；只想重畫圖時用 redraw()，不必重跑實驗）
def draw(summary, output):
    output = Path(output)
    label = summary.get('source_label', 'Reuters-21578')
    if summary['sweep']:
        limits = sorted({r['limit_mb'] for r in summary['sweep']})
        base = summary['methods'].get('baseline')
        reference = {'seconds': base['seconds'], 'peak_bytes': base['peak']} if base else None
        _line_panels(summary['sweep'], 'limit_mb', 'memory limit (MB)', output / 'memory_sweep.png',
                     f"Changing the memory limit (full {label}, {summary['dataset']['documents']:,} docs)",
                     [(v, f'{v:g}') for v in limits], reference=reference)
    if summary['scaling']:
        counts = sorted({r['documents'] for r in summary['scaling']})
        _line_panels(summary['scaling'], 'documents', 'documents', output / 'scaling.png',
                     f"Changing the number of documents (memory limit {summary['limit_mb']} MB; "
                     f"more than {summary['dataset']['documents']:,} = corpus read repeatedly)",
                     [(v, _short(v)) for v in counts], log_y=True, rotate=40)
    plot_stages(summary['methods'], output / 'stage_times.png', label)


def redraw(output=RESULTS_DIR):
    draw(json.loads((Path(output) / 'summary.json').read_text(encoding='utf-8')), output)


# ---------------------------------------------------------------------------
# 全部重跑
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    rows = list(rows)
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})


def _sample_json(samples):
    return json.loads(json.dumps(samples, default=list))


def run_all(output=None, limit_mb=DEFAULT_LIMIT_MB, repeats=5, sweep=SWEEP_MB, scales=None, source=DEFAULT_SOURCE):
    output = Path(output or results_dir(source))
    scales = scales_for(source) if scales is None else scales
    output.mkdir(parents=True, exist_ok=True)
    limit = limit_mb * MB
    started = perf_counter()

    _log('Part 1 · baseline inverted index')
    clock = perf_counter()
    reference = InvertedIndex(source=source)
    reference_seconds = perf_counter() - clock
    dataset = {'documents': reference.documents, 'tokens': reference.tokens, 'vocabulary': len(reference.index),
               'postings': reference.pairs(),
               'empty': reference.documents - len({d for p in reference.index.values() for d in p}),
               'baseline_seconds': reference_seconds}

    _log(f'Full corpus · memory limit {limit_mb} MB · {repeats} timed runs per method')
    full = full_comparison(limit, repeats, source)
    check = equivalence(reference, source)
    retrieval = retrieval_rows(source)
    _log(f"  identical to Part 1 index: baseline {check['baseline']}, BSBI {check['bsbi']}, SPIMI {check['spimi']}; "
         f"queries identical: {all(r['identical'] and r['baseline_identical'] for r in retrieval)}")

    _log('Memory limit sweep')
    sweep_rows = memory_sweep(sweep, source) if sweep else []
    _log('Scaling')
    scale_rows = scaling(scales, limit, source) if scales else []
    _clean(TMP_DIR / source)

    comparison = [{'metric': name, **{method: value(method) for method in METHODS}} for name, value in (
        ('Index construction time (s, median)', lambda m: full[m]['seconds']),
        ('  without preprocessing (s)', lambda m: full[m]['seconds'] - full[m]['times']['preprocess']),
        ('Memory usage (peak MB, tracemalloc)', lambda m: full[m]['peak'] / MB),
        ('Number of blocks', lambda m: full[m]['block_count']),
        ('Intermediate files', lambda m: full[m]['intermediate_files']),
        ('Intermediate files size (MB)', lambda m: full[m]['intermediate_bytes'] / MB),
        ('Final index size (MB)', lambda m: full[m]['final_bytes'] / MB),
        ('Entries held in blocks', lambda m: full[m]['block_entries']),
    )]
    _write_csv(output / 'comparison.csv', comparison)
    _write_csv(output / 'retrieval.csv', retrieval)
    if sweep_rows:
        _write_csv(output / 'memory_sweep.csv', sweep_rows)
    if scale_rows:
        _write_csv(output / 'scaling.csv', scale_rows)

    summary = {'source': source, 'source_label': get_source(source)['label'],
               'dataset': dataset, 'limit_mb': limit_mb, 'repeats': repeats, 'equivalence': check,
               'methods': {m: {k: v for k, v in r.items() if k != 'samples'} for m, r in full.items()},
               'samples': {m: _sample_json(r['samples']) for m, r in full.items()},
               'retrieval': retrieval, 'sweep': sweep_rows, 'scaling': scale_rows, 'heaps': heaps_fit(scale_rows, source),
               'complexity': complexity(), 'pair_bytes': bsbi.PAIR_BYTES,
               'spimi_bytes': {'entry': spimi.ENTRY_BYTES, 'posting': spimi.POSTING_BYTES},
               'seconds': perf_counter() - started}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    draw(summary, output)
    _log(f'Done in {summary["seconds"]:.0f} s → {output}')
    return summary


# 只重算檢索測試（改了 queries.py 之後用）：用 data/indexing/full/ 裡已經建好的三個最終索引重新查詢，
# 更新 retrieval.csv 與 summary.json 的 retrieval，其他實驗數字不動、不必重跑約 30 分鐘的實驗
def refresh_retrieval(output=None, source=DEFAULT_SOURCE):
    output = Path(output or results_dir(source))
    path = output / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['retrieval'] = retrieval_rows(source)
    _write_csv(output / 'retrieval.csv', summary['retrieval'])
    with open(path, 'w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary['retrieval']
