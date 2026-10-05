from django.core.management.base import BaseCommand, CommandError
from search.indexer import index_file
from search.pmc_client import download_pmc_xml, search_pmc


class Command(BaseCommand):
    help = "從 NCBI PMC E-utilities 下載 XML，可指定 PMCID 或用關鍵字搜尋後批次下載"

    def add_arguments(self, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--pmcid", type=str, help="例如 PMC8270360")
        group.add_argument("--query", type=str, help="PMC 搜尋關鍵字，例如 cancer immunotherapy")
        parser.add_argument("--limit", type=int, default=5, help="query 模式下載篇數，預設 5，最多 20")
        parser.add_argument("--output-dir", type=str, default="data/corpus")
        parser.add_argument("--email", type=str, default="student@example.com")
        parser.add_argument("--no-index", action="store_true", help="只下載 XML，不建立 Django 索引")

    def handle(self, *args, **options):
        ids = [options["pmcid"]] if options.get("pmcid") else search_pmc(
            options["query"], options["limit"], options["email"]
        )
        if not ids:
            raise CommandError("找不到符合條件的 PMC 文章")
        self.stdout.write(f"準備處理 {len(ids)} 篇：{', '.join(ids)}")
        for pmcid in ids:
            try:
                path = download_pmc_xml(pmcid, options["output_dir"], options["email"])
                msg = f"下載完成：{path}"
                if not options["no_index"]:
                    doc = index_file(path, replace=True)
                    msg += f"；已加入文章庫：{doc.title}"
                self.stdout.write(self.style.SUCCESS(msg))
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"{pmcid}: {exc}"))
