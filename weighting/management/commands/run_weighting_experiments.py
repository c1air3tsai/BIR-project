# =============================================================================
# python manage.py run_weighting_experiments
# Project 3 一鍵重跑全部實驗：重建索引 → 詞項統計 → RQ1/RQ2/RQ4 示範 → 評估 → 存 CSV 與 PNG
#
# 參數：
#   --output DIR        輸出資料夾（預設 docs/project3_results）
#   --keep-stopwords    保留 stopword（預設會移除）
#   --stemming          開啟 Porter stemming（預設不開）
# =============================================================================
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from weighting.evaluation import METRICS
from weighting.experiments import run_all
from weighting.index import load_index
from weighting.preprocessing import PreprocessConfig


class Command(BaseCommand):
    help = 'Run all Project 3 term-weighting experiments and write CSV / PNG results.'

    def add_arguments(self, parser):
        parser.add_argument('--output', default=str(Path(settings.BASE_DIR) / 'docs' / 'project3_results'))
        parser.add_argument('--keep-stopwords', action='store_true')
        parser.add_argument('--stemming', action='store_true')

    def handle(self, *args, **options):
        config = PreprocessConfig(remove_stopwords=not options['keep_stopwords'], stemming=options['stemming'])
        # force_rebuild=True：確保結果一定是用目前資料庫重新計算的
        index = load_index(config, force_rebuild=True)
        result = run_all(index, options['output'])
        summary = result['summary']
        self.stdout.write(f"Preprocessing: {config.label}")
        self.stdout.write(f"Documents {summary['documents']} · tokens {summary['tokens']} · "
                          f"vocabulary {summary['vocabulary']} · mean length {summary['avg_length']:.1f}")
        self.stdout.write('')
        self.stdout.write(f"{'Configuration':26s}" + ''.join(f'{name:>11s}' for name in METRICS.values()))
        for row in result['means']:
            self.stdout.write(f"{row['label']:26s}" + ''.join(f'{row[key]:11.3f}' for key in METRICS))
        self.stdout.write(self.style.SUCCESS(f"Results written to {options['output']}"))
