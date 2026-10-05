from django.core.management.base import BaseCommand
from search.topic_import import run_job


class Command(BaseCommand):
    help = 'Run one persistent topic import job (normally launched by Upload).'

    def add_arguments(self, parser):
        parser.add_argument('job_id')

    def handle(self, *args, **options):
        run_job(options['job_id'])
