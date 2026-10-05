from django.contrib import admin
from .models import Document, Posting, Term


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "pmcid", "pmid", "journal", "publication_year", "word_count", "indexed_at")
    search_fields = ("title", "pmcid", "pmid", "doi", "journal", "authors")
    readonly_fields = ("char_count", "word_count", "sentence_count", "avg_words_per_sentence", "indexed_at")


@admin.register(Term)
class TermAdmin(admin.ModelAdmin):
    search_fields = ("word",)


@admin.register(Posting)
class PostingAdmin(admin.ModelAdmin):
    list_display = ("term", "document", "term_freq")
    search_fields = ("term__word", "document__title")
