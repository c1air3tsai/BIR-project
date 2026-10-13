from django.core.management.base import BaseCommand, CommandError

from search import embeddings
from search.models import Document


class Command(BaseCommand):
    help = "Build or validate the persistent GLP-1, Balanced, and CS Word2Vec preset caches."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Rebuild both models and every global visualization even when the manifest is valid.",
        )

    def handle(self, *args, **options):
        if not Document.objects.exclude(abstract="").exists():
            raise CommandError("No abstracts are available. Import the Medical and CS corpora first.")
        force = options["force"]
        for preset in ("medical", "balanced", "cs"):
            result = embeddings.precompute_preset(Document.objects.all(), preset, force=force)
            selection = result["selection"]
            manifest = result["manifest"]
            status = "CACHE HIT / SKIP" if result["cache_hit"] else "BUILT"
            models = manifest.get("models", {})
            timings = manifest.get("timings", {})
            self.stdout.write(self.style.SUCCESS(
                f"{status}: {embeddings.PRESET_SPECS[preset]['label']} "
                f"({selection['actual']['medical']} Medical + {selection['actual']['cs']} CS) "
                f"CBOW={models.get('cbow', {}).get('training_time_seconds', 'n/a')}s/"
                f"{models.get('cbow', {}).get('vocabulary_size', 'n/a')} terms, "
                f"Skip-Gram={models.get('skipgram', {}).get('training_time_seconds', 'n/a')}s/"
                f"{models.get('skipgram', {}).get('vocabulary_size', 'n/a')} terms, "
                f"PCA/t-SNE={timings.get('artifact_total_seconds', 'n/a')}s, "
                f"cache={embeddings.cache_directory(selection['records'], embeddings.DEFAULTS, result['cache_name'])}"
            ))
