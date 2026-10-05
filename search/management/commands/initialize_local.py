"""Report persistent collection state without installing sample articles."""
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from search.models import Document, Term


BUNDLED_SAMPLE_FILES = {
    '39932906.xml', '39961847.xml', '40145494.xml', '40175414.xml',
    '40332938.xml', '40335893.xml', '40605234.xml', '40760693.xml',
    '41095831.xml', '41364434.xml', 'PMC10009402.xml', 'PMC12503546.xml',
    'PMC13559554.xml', 'PMC1831737.xml',
}


class Command(BaseCommand):
    help = 'Keep the persistent collection as-is; no sample articles are installed.'

    def handle(self, *args, **options):
        samples = Document.objects.filter(source_file__in=BUNDLED_SAMPLE_FILES)
        removed = samples.count()
        samples.delete()
        Term.objects.filter(postings__isnull=True).delete()
        corpus = Path(settings.BASE_DIR) / 'data' / 'corpus'
        for name in BUNDLED_SAMPLE_FILES:
            (corpus / name).unlink(missing_ok=True)
        if removed:
            self.stdout.write(f'Removed {removed} bundled sample articles.')
        self.stdout.write(f'Persistent collection ready: {Document.objects.count()} articles retained.')
