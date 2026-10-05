import re
from pathlib import Path

from django import forms

from .models import Topic


def get_or_create_topic(name):
    name = " ".join((name or "").split())
    if not name:
        return None
    topic, _ = Topic.objects.get_or_create(key=name.casefold(), defaults={"name": name})
    return topic


class TopicImportForm(forms.Form):
    source = forms.ChoiceField(choices=[('pubmed','PubMed · Medical'),('arxiv','arXiv · Computer Science')], required=False, initial='pubmed', label='Article source')
    query = forms.CharField(max_length=120, label="Topic or PubMed query",
        widget=forms.TextInput(attrs={"placeholder": "e.g. GLP-1", "autocomplete": "off"}))
    count = forms.IntegerField(min_value=1, max_value=1000, initial=1000,
        label="Articles to add", widget=forms.NumberInput(attrs={"min": 1, "max": 1000}))

    def clean_source(self):
        return self.cleaned_data['source'] or 'pubmed'

    def clean_query(self):
        query = " ".join(self.cleaned_data["query"].split())
        if not query:
            raise forms.ValidationError("Enter a topic or PubMed query.")
        return query


class TopicLabelForm(forms.Form):
    topic = forms.CharField(max_length=120, required=False, label="Topic (optional)",
        widget=forms.TextInput(attrs={"placeholder": "e.g. GLP-1", "list": "topic-names"}))


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    """Validate every file selected by one multi-file input."""

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            return [single_file_clean(item, initial) for item in data]
        return [single_file_clean(data, initial)]


class UploadDocumentForm(TopicLabelForm):
    files = MultipleFileField(
        label="XML files",
        help_text="Select up to 20 PubMed / PubMed Central XML articles.",
        widget=MultipleFileInput(attrs={"accept": ".xml", "multiple": True}),
    )

    def clean_files(self):
        files = self.cleaned_data["files"]
        if len(files) > 20:
            raise forms.ValidationError("Please upload no more than 20 XML files at one time.")
        invalid_names = [
            Path(uploaded.name).name
            for uploaded in files
            if Path(uploaded.name).suffix.lower() != ".xml"
        ]
        if invalid_names:
            raise forms.ValidationError(
                "Only .xml files are supported: " + ", ".join(invalid_names)
            )
        return files


class ArticleFetchForm(TopicLabelForm):
    identifiers = forms.CharField(
        label="PMID or PMCID(s)",
        help_text=(
            "Enter one or more IDs. Use the PMC prefix for a PMCID; "
            "plain numbers are treated as PMID. Separate IDs with commas, spaces, or new lines."
        ),
        widget=forms.Textarea(attrs={
            "placeholder": "42724776\nPMC12503546",
            "rows": 5,
        }),
    )

    def clean_identifiers(self):
        raw = self.cleaned_data["identifiers"]
        values = [x.strip() for x in re.split(r"[\s,;]+", raw) if x.strip()]
        # Preserve order while removing repeated IDs within the same request.
        values = list(dict.fromkeys(values))
        if not values:
            raise forms.ValidationError("Please enter at least one PMID or PMCID.")
        if len(values) > 20:
            raise forms.ValidationError("Please fetch no more than 20 IDs at one time.")
        return values
