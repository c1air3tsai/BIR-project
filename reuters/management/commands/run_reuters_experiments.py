# =============================================================================
# python manage.py run_reuters_experiments
# Project 4 一鍵重跑：下載並解析 Reuters-21578（若尚未下載）→ 訓練 / 讀取 Word2Vec →
# 詞層級分析 → 文件層級檢索評估 → 輸出 CSV / PNG 到 docs/project4_results/
#
# 參數：
#   --output DIR            輸出資料夾（預設 docs/project4_results）
#   --skip-corpus-size      不做「train + unused 語料」的額外實驗（省下約 3 分鐘訓練時間）
# =============================================================================
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from reuters.corpus import build_cache
from reuters.evaluation import METRICS
from reuters.experiments import run_all


class Command(BaseCommand):
    help = 'Run all Project 4 Reuters experiments and write CSV / PNG results.'

    def add_arguments(self, parser):
        parser.add_argument('--output', default=str(Path(settings.BASE_DIR) / 'docs' / 'project4_results'))
        parser.add_argument('--skip-corpus-size', action='store_true')

    def handle(self, *args, **options):
        build_cache()
        result = run_all(options['output'], include_corpus_size=not options['skip_corpus_size'])
        data = result['dataset']
        self.stdout.write(f"Test collection {data['documents']} docs · vocabulary {data['vocabulary']} · "
                          f"topics {data['topic_counts']}")
        self.stdout.write('')
        self.stdout.write(f"{'Method':34s}" + ''.join(f'{name:>11s}' for name in METRICS.values()))
        for row in result['means']:
            self.stdout.write(f"{row['label']:34s}" + ''.join(f'{row[key]:11.3f}' for key in METRICS))
        for row in result['corpus_size']:
            self.stdout.write(f"  corpus={row['w2v_corpus']:13s} {row['method']:16s} nDCG@10 {row['ndcg10']:.3f}")
        self.stdout.write(self.style.SUCCESS(f"Results written to {options['output']}"))
