from django.core.management.base import BaseCommand

from search.term_weighting import build_index


class Command(BaseCommand):
    help = 'Build or validate persistent sparse retrieval indices.'

    def add_arguments(self, parser):
        parser.add_argument('--force', action='store_true', help='Rebuild even if the current cache is valid')
        parser.add_argument('--corpus', choices=['medical', 'cs', 'all'], default='medical')

    def handle(self, *args, **options):
        result = build_index(options['corpus'], force=options['force'])
        self.stdout.write(self.style.SUCCESS(
            f"{'CACHE HIT' if result['cache_hit'] else 'BUILT'} {options['corpus']}: "
            f"{result['indexed']}/{result['available']} documents, {result['vocabulary']} terms, "
            f"key={result['key'][:12]}, {result['seconds']:.3f}s"))
