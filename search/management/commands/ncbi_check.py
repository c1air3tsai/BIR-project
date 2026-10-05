"""A small end-to-end network check using the same trusted HTTPS client as imports."""
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from search.pmc_client import search_pubmed, fetch_pubmed_batch


class Command(BaseCommand):
    help = 'Check trusted HTTPS, PubMed search and one abstract download.'

    def handle(self, *args, **options):
        directory = Path(settings.NCBI_CA_DIR)
        count = len({*directory.glob('*.crt'), *directory.glob('*.pem')})
        self.stdout.write(f'HTTPS uses the system trust store + {count} additional CA files.')
        try:
            ids, available = search_pubmed('GLP-1', retmax=1)
            if not ids:
                raise RuntimeError('PubMed returned no IDs for the test query.')
            records = fetch_pubmed_batch(ids[:1])
            if not records:
                raise RuntimeError('PubMed did not return a usable XML record.')
        except Exception as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f'NCBI OK: {available} search matches; downloaded PMID {records[0][0]}.'))
