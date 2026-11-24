# =============================================================================
# python manage.py run_indexing_experiments
# Project 5 一鍵重跑：Part 1 基準索引 → BSBI / SPIMI（完整語料，計時 + 記憶體 + 中間結果）→
# 逐一比對兩個索引 → 5 組查詢的檢索測試 → 改變記憶體上限 → 改變文件數（到 1,000,000 篇）
# 輸出到 docs/project5_results/（PubMed 在 docs/project5_results/pubmed/）；
# 最終索引留在 data/indexing/full/（PubMed 在 data/indexing/pubmed/full/），網頁即時查詢用
#
# 參數：
#   --source NAME     資料來源：reuters（預設，Reuters-21578）或 pubmed（PubMed GLP-1，先跑 fetch_pubmed_glp1）
#   --memory-mb N     記憶體上限（預設 4 MB，兩種方法相同）
#   --repeats N       計時重複次數（預設 5，取中位數）
#   --skip-sweep      不做「改變記憶體上限」
#   --max-docs N      規模實驗只跑到 N 篇（預設跑到 1,000,000 篇；Reuters 全部約 30 分鐘）
#   --redraw          不重跑實驗，只用現有的 summary.json 重畫 PNG
#   --retrieval-only  不重跑實驗，只用已建好的最終索引重算檢索測試（改了 queries.py 之後用）
# =============================================================================
from django.core.management.base import BaseCommand

from indexing.experiments import (DEFAULT_LIMIT_MB, SCALES, SWEEP_MB, redraw, refresh_retrieval, results_dir, run_all,
                                  scales_for)
from indexing.sources import SOURCES
from reuters.corpus import build_cache


class Command(BaseCommand):
    help = 'Run all Project 5 experiments (BSBI vs SPIMI) and write CSV / PNG / JSON results.'

    def add_arguments(self, parser):
        parser.add_argument('--source', choices=list(SOURCES), default='reuters')
        parser.add_argument('--output', default=None)
        parser.add_argument('--memory-mb', type=int, default=DEFAULT_LIMIT_MB)
        parser.add_argument('--repeats', type=int, default=5)
        parser.add_argument('--skip-sweep', action='store_true')
        parser.add_argument('--max-docs', type=int, default=max(SCALES))
        parser.add_argument('--redraw', action='store_true')
        parser.add_argument('--retrieval-only', action='store_true')

    def handle(self, *args, **options):
        source = options['source']
        output = options['output'] or results_dir(source)
        if options['redraw']:
            redraw(output)
            self.stdout.write(self.style.SUCCESS(f'Charts redrawn in {output}'))
            return
        if options['retrieval_only']:
            rows = refresh_retrieval(output, source)
            same = all(row['identical'] and row['baseline_identical'] for row in rows)
            self.stdout.write(self.style.SUCCESS(f'{len(rows)} retrieval rows, all identical: {same}'))
            return
        if source == 'reuters':
            build_cache()
        scales = tuple(n for n in scales_for(source) if n <= options['max_docs'])
        summary = run_all(output, options['memory_mb'], options['repeats'],
                          () if options['skip_sweep'] else SWEEP_MB, scales, source)
        for row in summary['retrieval']:
            if not row['identical']:
                self.stdout.write(self.style.ERROR(f"Mismatch: {row['query']} {row['expression']}"))
        self.stdout.write(self.style.SUCCESS(f'Results written to {output}'))
