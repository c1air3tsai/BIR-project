import uuid
from django.db import models


class Topic(models.Model):
    name = models.CharField(max_length=120)
    key = models.CharField(max_length=120, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Document(models.Model):
    title = models.CharField(max_length=500)
    source_file = models.CharField(max_length=300, unique=True)
    pmcid = models.CharField(max_length=40, blank=True, db_index=True)
    pmid = models.CharField(max_length=40, blank=True, db_index=True)
    arxiv_id = models.CharField(max_length=50, blank=True, db_index=True)
    doi = models.CharField(max_length=200, blank=True)
    journal = models.CharField(max_length=300, blank=True)
    publication_year = models.CharField(max_length=10, blank=True)
    authors = models.TextField(blank=True)
    abstract = models.TextField(blank=True)
    raw_text = models.TextField()
    char_count = models.IntegerField(default=0)
    word_count = models.IntegerField(default=0)
    sentence_count = models.IntegerField(default=0)
    avg_words_per_sentence = models.FloatField(default=0)
    indexed_at = models.DateTimeField(auto_now=True)
    topics = models.ManyToManyField(Topic, blank=True, related_name="documents")

    class Meta:
        ordering = ["-indexed_at", "title"]

    def __str__(self):
        return self.title


class Term(models.Model):
    word = models.CharField(max_length=100, unique=True, db_index=True)

    def __str__(self):
        return self.word


class Posting(models.Model):
    term = models.ForeignKey(Term, on_delete=models.CASCADE, related_name="postings")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="postings")
    term_freq = models.IntegerField(default=1)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["term", "document"], name="unique_term_document")
        ]
        indexes = [models.Index(fields=["term", "document"])]

    def __str__(self):
        return f"{self.term.word} -> doc#{self.document_id} (tf={self.term_freq})"


class ImportJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name="imports")
    source = models.CharField(max_length=20, default='pubmed', choices=[('pubmed','PubMed'),('arxiv','arXiv CS')])
    requested = models.PositiveIntegerField()
    status = models.CharField(max_length=20, default="queued")
    imported = models.PositiveIntegerField(default=0)
    linked = models.PositiveIntegerField(default=0)
    duplicates = models.PositiveIntegerField(default=0)
    skipped = models.PositiveIntegerField(default=0)
    examined = models.PositiveIntegerField(default=0)
    available = models.PositiveIntegerField(default=0)
    offset = models.PositiveIntegerField(default=0)
    message = models.TextField(blank=True)
    cancel_requested = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(
            models.Value(1), condition=models.Q(status__in=["queued", "running"]),
            name="one_active_topic_import",
        )]

    @property
    def added(self):
        return self.imported + self.linked

    @property
    def progress(self):
        return min(100, round(100 * self.added / self.requested)) if self.requested else 0

    @property
    def is_active(self):
        return self.status in {"queued", "running"}
