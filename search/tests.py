import tempfile
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .forms import UploadDocumentForm
from .indexer import parse_document
from .models import Document
from .text_processing import document_stats, preprocess, split_sentences, tokenize
from .pmc_client import normalize_identifier


class TextProcessingTests(TestCase):
    def test_preprocess(self):
        self.assertIn("cancer", preprocess("The cancers are growing."))

    def test_biomedical_tokenization(self):
        tokens = tokenize("COVID-19 SARS-CoV-2 IL-6 patient's e.g. 48.1% 0.05")
        self.assertEqual(
            tokens,
            ["covid-19", "sars-cov-2", "il-6", "patient's", "e.g.", "48.1%", "0.05"],
        )

    def test_word_separator_rules(self):
        self.assertEqual(tokenize("activity/exercise"), ["activity", "exercise"])
        self.assertEqual(tokenize("0.44–0.97"), ["0.44", "0.97"])

    def test_project_word_count_examples(self):
        cases = {
            "0.65": ["0.65"],
            "24.9%": ["24.9%"],
            "0.65.": ["0.65"],
            "12-15%": ["12", "15%"],
            "12–15%": ["12", "15%"],
            "OR 0.65, CI: 0.44–0.97": ["or", "0.65", "ci", "0.44", "0.97"],
            "5-stage 5‐stage 5‑stage 5–stage": [
                "5-stage", "5‐stage", "5‑stage", "5–stage",
            ],
            "HIV-1 SARS-CoV-2 age‐specific non–small": [
                "hiv-1", "sars-cov-2", "age‐specific", "non–small",
            ],
            "Alzheimer's O’Malley factors—particularly": [
                "alzheimer's", "o’malley", "factors", "particularly",
            ],
            "i.e. e.g. U.S. Δ β α Brca1ΔC/ΔC": [
                "i.e.", "e.g.", "u.s.", "δ", "β", "α", "brca1δc", "δc",
            ],
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(tokenize(source), expected)

    def test_sentence_rules(self):
        text = "Dr. Smith measured 3.14 mg. The result was significant."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_multidot_abbreviation(self):
        text = "Several factors, e.g. age and sex, were recorded. Results were stable."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_lowercase_sentence_start_is_not_lost(self):
        text = "The pathway was activated. p53 expression then increased."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_stats(self):
        stats = document_stats("COVID-19 treatment works. It helps patients.")
        self.assertEqual(stats["sentence_count"], 2)
        self.assertEqual(stats["word_count"], 6)

    def test_whitespace_and_keyword_line_stats(self):
        stats = document_stats("risk\n\nassessment\nKeywords: heart disease")
        self.assertEqual(stats["char_count"], len("risk assessment Keywords: heart disease"))
        self.assertEqual(stats["char_count_no_space"], len("riskassessmentKeywords:heartdisease"))
        self.assertEqual(stats["sentence_count"], 3)

    def test_chinese_sentence_endings(self):
        self.assertEqual(len(split_sentences("第一句。\n第二句！\n第三句？")), 3)


class XmlExtractionTests(SimpleTestCase):
    def _parse_xml(self, xml):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "article.xml"
            path.write_text(xml, encoding="utf-8")
            return parse_document(str(path))

    def test_jats_main_abstract_headings_and_keywords(self):
        title, search_text, meta = self._parse_xml("""
            <article><front><article-meta>
              <title-group><article-title>Not counted or searched</article-title></title-group>
              <abstract abstract-type="toc"><p>Auxiliary summary.</p></abstract>
              <abstract abstract-type="normal">
                <sec><title>Results</title><p>Main result text.</p></sec>
              </abstract>
              <kwd-group><title>Keywords</title><kwd>heart disease</kwd><kwd>β cell</kwd></kwd-group>
            </article-meta></front><body><sec><title>Introduction</title><p>Body text.</p></sec></body></article>
        """)
        self.assertEqual(title, "Not counted or searched")
        self.assertEqual(meta["abstract"], "Main result text.\n\nKeywords: heart disease, β cell")
        self.assertNotIn("Results", meta["abstract"])
        self.assertNotIn(title, search_text)
        self.assertIn("Body text.", search_text)

    def test_existing_keywords_line_stops_abstract(self):
        _title, _search_text, meta = self._parse_xml("""
            <article><front><article-meta>
              <title-group><article-title>Title</article-title></title-group>
              <abstract>
                <p>Abstract content.</p>
                <p>Keywords: existing, terms</p>
                <p>Reply article text must be excluded.</p>
              </abstract>
              <kwd-group><kwd>duplicate metadata term</kwd></kwd-group>
            </article-meta></front></article>
        """)
        self.assertEqual(meta["abstract"], "Abstract content.\n\nKeywords: existing, terms")

    def test_pubmed_abstract_and_keywords(self):
        title, search_text, meta = self._parse_xml("""
            <PubmedArticleSet><PubmedArticle><MedlineCitation>
              <PMID>1</PMID><Article>
                <ArticleTitle>Bibliographic title</ArticleTitle>
                <Abstract>
                  <AbstractText Label="BACKGROUND">First block.</AbstractText>
                  <AbstractText Label="RESULTS">Second block.</AbstractText>
                </Abstract>
              </Article>
              <KeywordList><Keyword>one</Keyword><Keyword>two</Keyword></KeywordList>
            </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """)
        self.assertEqual(meta["abstract"], "First block.\n\nSecond block.\n\nKeywords: one, two")
        self.assertEqual(search_text, meta["abstract"])
        self.assertNotIn(title, search_text)


class IdentifierTests(TestCase):
    def test_identifier_normalization(self):
        self.assertEqual(normalize_identifier("PMC12503546"), ("pmc", "PMC12503546"))
        self.assertEqual(normalize_identifier("42724776"), ("pubmed", "42724776"))
        self.assertEqual(normalize_identifier("PMID:42724776"), ("pubmed", "42724776"))


class MultipleXmlUploadTests(TestCase):
    @staticmethod
    def _xml_file(name, pmid, title):
        xml = f"""
            <PubmedArticleSet><PubmedArticle><MedlineCitation>
              <PMID>{pmid}</PMID><Article>
                <ArticleTitle>{title}</ArticleTitle>
                <Abstract><AbstractText>Abstract for {title}.</AbstractText></Abstract>
              </Article>
            </MedlineCitation></PubmedArticle></PubmedArticleSet>
        """.encode("utf-8")
        return SimpleUploadedFile(name, xml, content_type="application/xml")

    def test_upload_form_accepts_multiple_xml_files(self):
        form = UploadDocumentForm(
            data={},
            files={"files": [
                self._xml_file("one.xml", "101", "First article"),
                self._xml_file("two.xml", "102", "Second article"),
            ]},
        )
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(len(form.cleaned_data["files"]), 2)

    def test_upload_view_indexes_all_selected_files(self):
        with tempfile.TemporaryDirectory() as directory, self.settings(BASE_DIR=Path(directory)):
            response = self.client.post(
                reverse("search:import"),
                data={
                    "action": "upload",
                    "files": [
                        self._xml_file("one.xml", "201", "First uploaded article"),
                        self._xml_file("two.xml", "202", "Second uploaded article"),
                    ],
                },
            )

            self.assertRedirects(response, reverse("search:import"), fetch_redirect_response=False)
            self.assertEqual(Document.objects.count(), 2)
            self.assertSetEqual(
                set(Document.objects.values_list("title", flat=True)),
                {"First uploaded article", "Second uploaded article"},
            )
            self.assertEqual(len(list((Path(directory) / "data" / "corpus").glob("*.xml"))), 2)

    def test_batch_upload_keeps_successes_when_one_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory, self.settings(BASE_DIR=Path(directory)):
            response = self.client.post(
                reverse("search:import"),
                data={
                    "action": "upload",
                    "files": [
                        self._xml_file("valid.xml", "301", "Valid article"),
                        SimpleUploadedFile(
                            "broken.xml", b"<not-valid", content_type="application/xml"
                        ),
                    ],
                },
            )

            self.assertEqual(response.status_code, 302)
            self.assertEqual(Document.objects.count(), 1)
            result = self.client.session["import_result"]
            self.assertEqual(result["kind"], "partial")
            self.assertEqual(len(result["uploaded"]), 1)
            self.assertEqual(len(result["errors"]), 1)
