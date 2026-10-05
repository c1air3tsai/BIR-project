"""Report the running revision, SQLite mode and most recent import failure."""
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import connection
from search.models import ImportJob


class Command(BaseCommand):
    help = 'Show revision, actual SQLite journal mode and recent worker logs.'

    def handle(self, *args, **options):
        self.stdout.write(f'Revision: {settings.APP_REVISION}')
        with connection.cursor() as cursor:
            self.stdout.write(f'SQLite journal: {cursor.execute("PRAGMA journal_mode").fetchone()[0]}')
            self.stdout.write(f'Busy timeout: {cursor.execute("PRAGMA busy_timeout").fetchone()[0]} ms')
        self.stdout.write(f'Transaction mode: {connection.settings_dict["OPTIONS"].get("transaction_mode")}')
        for job in ImportJob.objects.select_related('topic')[:3]:
            self.stdout.write(f'{job.topic.name} [{job.get_source_display()}]: {job.status}; {job.added}/{job.requested}; examined {job.examined}; skipped {job.skipped}; offset {job.offset}')
            self.stdout.write(job.message)
            log = Path(settings.BASE_DIR) / 'data' / 'logs' / f'import-{job.pk}.log'
            if log.is_file():
                self.stdout.write('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-25:]))
