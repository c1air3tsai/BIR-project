"""Initialize a persistent container database without clearing existing articles."""
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import timezone
from search.models import ImportJob


class Command(BaseCommand):
    help = 'Retain persistent articles and recover jobs interrupted by a container restart.'

    def handle(self, *args, **options):
        call_command('initialize_local', verbosity=options['verbosity'])
        recovered = ImportJob.objects.filter(status__in=['queued', 'running']).update(
            status='interrupted', updated_at=timezone.now(),
            message='The container restarted. Collected articles were kept. Repeat the topic import to add the remaining articles.',
        )
        if recovered:
            self.stdout.write(f'Recovered {recovered} interrupted import job(s); saved articles retained.')
