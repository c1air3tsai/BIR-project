# search/management/commands/build_index.py
# 讓我們可以在終端機下指令：python manage.py build_index
# Django 會自動找 management/commands/ 底下的檔案，把檔名當成指令名稱。

from django.core.management.base import BaseCommand
from search.indexer import build_index


class Command(BaseCommand):
    help = "讀取 data/corpus 底下的文件，建立 inverted index 並存入資料庫"

    def add_arguments(self, parser):
        parser.add_argument(
            "--corpus-dir",
            type=str,
            default="data/corpus",
            help="語料庫資料夾路徑（預設 data/corpus）",
        )

    def handle(self, *args, **options):
        corpus_dir = options["corpus_dir"]
        self.stdout.write(f"開始從 {corpus_dir} 建立索引...")
        num_docs, num_terms = build_index(corpus_dir)
        self.stdout.write(self.style.SUCCESS(
            f"完成！共索引 {num_docs} 篇文件，{num_terms} 個不重複詞彙。"
        ))
