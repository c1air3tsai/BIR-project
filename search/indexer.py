import os
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from django.db import transaction

from .models import Document, Posting, Term
from .query_expansion import clear_variant_cache
from .text_processing import document_stats, preprocess


def _first_text(root, xpath):
    el = root.find(xpath)
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _article_id(root, kind):
    el = root.find(f".//article-id[@pub-id-type='{kind}']")
    return (el.text or "").strip() if el is not None else ""


def _local_name(tag):
    return str(tag).split("}")[-1]


def _clean_node_text(node):
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


_KEYWORDS_LABEL = re.compile(r"\bkeywords?\s*:", flags=re.IGNORECASE)


def _append_keywords(abstract, keywords):
    """Append one canonical Keywords line unless the abstract already has one."""
    abstract = (abstract or "").strip()
    keywords = [keyword.strip() for keyword in keywords if keyword and keyword.strip()]
    if not keywords or _KEYWORDS_LABEL.search(abstract):
        return abstract
    keyword_line = f"Keywords: {', '.join(keywords)}"
    return "\n\n".join(part for part in [abstract, keyword_line] if part)


def _truncate_abstract_parts_at_keywords(parts):
    """Keep an existing Keywords block, but discard abstract blocks after it."""
    kept = []
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        kept.append(part)
        if _KEYWORDS_LABEL.search(part):
            break
    return kept


def _jats_keywords(article_meta):
    if article_meta is None:
        return []
    keywords = []
    for group in article_meta.iter():
        if _local_name(group.tag) != "kwd-group":
            continue
        for node in group.iter():
            if _local_name(node.tag) == "kwd":
                text = _clean_node_text(node)
                if text:
                    keywords.append(text)
    return keywords


def _extract_abstract(root):
    """
    Extract only the article's main PMC/JATS abstract.

    PMC can contain several <abstract> elements, for example a normal abstract,
    abstract-type="toc", and abstract-type="editor".  For the assignment we
    want the formal article abstract only, not TOC/editor summaries.

    Section headings such as "Background", "Methods and Findings", and
    "Conclusions" are intentionally NOT included in the returned text, so they
    do not affect word/character/sentence statistics.
    """
    article_meta = next(
        (node for node in root.iter() if _local_name(node.tag) == "article-meta"),
        None,
    )
    if article_meta is None:
        return ""

    abstracts = [
        node for node in article_meta.iter()
        if _local_name(node.tag) == "abstract"
    ]
    if not abstracts:
        return _append_keywords("", _jats_keywords(article_meta))

    # First choice: the standard JATS main abstract normally has no
    # abstract-type attribute.  This is the case for PMC1831737.
    abstract = next(
        (
            node for node in abstracts
            if not (node.attrib.get("abstract-type", "") or "").strip()
        ),
        None,
    )

    # Second choice: journals that explicitly label the main abstract normal.
    if abstract is None:
        abstract = next(
            (
                node for node in abstracts
                if (node.attrib.get("abstract-type", "") or "").strip().lower()
                == "normal"
            ),
            None,
        )

    # Final fallback for sources using a non-standard main-abstract type.
    if abstract is None:
        excluded_types = {
            "toc", "editor", "graphical", "teaser", "short", "lay",
            "plain-language-summary", "lay-summary", "layperson",
        }
        abstract = next(
            (
                node for node in abstracts
                if (node.attrib.get("abstract-type", "") or "").strip().lower()
                not in excluded_types
            ),
            None,
        )
    if abstract is None:
        return _append_keywords("", _jats_keywords(article_meta))

    # Count/display only abstract paragraph text.  Do not include <title>
    # elements such as Background / Methods and Findings / Conclusions.
    paragraphs = []
    keyword_heading_pending = False
    for node in abstract.iter():
        tag = _local_name(node.tag)
        if tag == "title":
            heading = _clean_node_text(node)
            keyword_heading_pending = bool(
                re.fullmatch(r"keywords?\s*:?\s*", heading, flags=re.IGNORECASE)
            )
            continue
        if tag != "p":
            continue
        text = _clean_node_text(node)
        if text:
            if keyword_heading_pending and not _KEYWORDS_LABEL.search(text):
                text = f"Keywords: {text}"
            paragraphs.append(text)
            if keyword_heading_pending or _KEYWORDS_LABEL.search(text):
                break
        keyword_heading_pending = False

    if paragraphs:
        abstract_text = "\n\n".join(_truncate_abstract_parts_at_keywords(paragraphs))
        return _append_keywords(abstract_text, _jats_keywords(article_meta))

    # Rare unstructured abstract with no <p>: rebuild text while excluding
    # nested <title> elements so labels still do not enter statistics.
    pieces = []
    if abstract.text and abstract.text.strip():
        pieces.append(abstract.text.strip())
    for child in abstract:
        if _local_name(child.tag) == "title":
            if child.tail and child.tail.strip():
                pieces.append(child.tail.strip())
            continue
        child_text = _clean_node_text(child)
        if child_text:
            pieces.append(child_text)
        if child.tail and child.tail.strip():
            pieces.append(child.tail.strip())
    abstract_text = " ".join(_truncate_abstract_parts_at_keywords(pieces)).strip()
    return _append_keywords(abstract_text, _jats_keywords(article_meta))


def _body_text(root):
    """Keep PMC/JATS section titles and paragraphs separated for future full-text use."""
    body = next((node for node in root.iter() if _local_name(node.tag) == "body"), None)
    if body is None:
        return ""

    blocks = []
    for node in body.iter():
        tag = _local_name(node.tag)
        if tag not in {"title", "p"}:
            continue
        text = _clean_node_text(node)
        if text:
            blocks.append(text)
    return "\n\n".join(blocks)


def _extract_jats_xml(root, path):
    title = _first_text(root, ".//article-title") or os.path.basename(path)
    abstract = _extract_abstract(root)
    body = _body_text(root) or _first_text(root, ".//body")
    # Search scope excludes the bibliographic title but includes the supplied
    # abstract (with Keywords) and PMC body.
    full_text = "\n\n".join(x for x in [abstract, body] if x).strip()

    author_names = []
    for contrib in root.findall(".//contrib[@contrib-type='author']"):
        surname = _first_text(contrib, ".//surname")
        given = _first_text(contrib, ".//given-names")
        name = " ".join(x for x in [given, surname] if x)
        if name:
            author_names.append(name)

    # Some PMC/JATS records also carry the PMID in article-id.
    pmid = _article_id(root, "pmid")

    meta = {
        "pmcid": _article_id(root, "pmcid") or _article_id(root, "pmcaid") or _article_id(root, "pmc"),
        "pmid": pmid,
        "doi": _article_id(root, "doi"),
        "journal": _first_text(root, ".//journal-title"),
        "publication_year": _first_text(root, ".//pub-date/year") or _first_text(root, ".//year"),
        "authors": ", ".join(author_names),
        "abstract": abstract,
    }
    return title, full_text, meta


def _find_first_local(root, name):
    return next((node for node in root.iter() if _local_name(node.tag) == name), None)


def _find_all_local(root, name):
    return [node for node in root.iter() if _local_name(node.tag) == name]


def _pubmed_article_id(root, id_type):
    wanted = id_type.lower()
    for node in _find_all_local(root, "ArticleId"):
        node_type = (node.attrib.get("IdType", "") or "").lower()
        if node_type == wanted:
            return (node.text or "").strip()
    return ""


def _extract_pubmed_abstract(article):
    """Extract PubMed AbstractText elements and preserve structured-abstract blocks."""
    abstract_nodes = _find_all_local(article, "AbstractText")
    parts = []
    for node in abstract_nodes:
        text = _clean_node_text(node)
        if text:
            parts.append(text)
    parts = _truncate_abstract_parts_at_keywords(parts)

    medline = next(
        (node for node in article.iter() if _local_name(node.tag) == "MedlineCitation"),
        None,
    )
    keywords = []
    if medline is not None:
        for keyword_list in medline.iter():
            if _local_name(keyword_list.tag) != "KeywordList":
                continue
            for node in keyword_list.iter():
                if _local_name(node.tag) == "Keyword":
                    text = _clean_node_text(node)
                    if text:
                        keywords.append(text)

    return _append_keywords("\n\n".join(parts), keywords)


def _extract_pubmed_year(article):
    # Prefer explicit Year fields in article dates / journal issue dates.
    for parent_name in ("ArticleDate", "PubDate", "DateCompleted", "DateRevised"):
        for parent in _find_all_local(article, parent_name):
            year_node = next((n for n in parent.iter() if _local_name(n.tag) == "Year"), None)
            if year_node is not None and (year_node.text or "").strip():
                return (year_node.text or "").strip()

    # MedlineDate can be values such as "2025 Jan-Feb".
    medline_date = _find_first_local(article, "MedlineDate")
    if medline_date is not None:
        match = re.search(r"\b(19|20)\d{2}\b", medline_date.text or "")
        if match:
            return match.group(0)
    return ""


def _extract_pubmed_xml(root, path):
    """Parse PubMed EFetch XML (PubmedArticleSet/PubmedArticle)."""
    article = _find_first_local(root, "PubmedArticle")
    if article is None:
        raise ValueError("This XML does not contain a PubMed article")

    title_node = _find_first_local(article, "ArticleTitle")
    title = _clean_node_text(title_node) or os.path.basename(path)
    abstract = _extract_pubmed_abstract(article)

    # PubMed XML has no body, and the bibliographic title is not searchable.
    full_text = abstract

    pmid_node = _find_first_local(article, "PMID")
    pmid = (pmid_node.text or "").strip() if pmid_node is not None else ""
    pmcid = _pubmed_article_id(article, "pmc")
    doi = _pubmed_article_id(article, "doi")

    journal_node = _find_first_local(article, "Journal")
    journal = ""
    if journal_node is not None:
        journal_title = next((n for n in journal_node.iter() if _local_name(n.tag) == "Title"), None)
        journal = _clean_node_text(journal_title)

    author_names = []
    author_list = _find_first_local(article, "AuthorList")
    if author_list is not None:
        for author in author_list:
            if _local_name(author.tag) != "Author":
                continue
            fore = next((n for n in author.iter() if _local_name(n.tag) == "ForeName"), None)
            last = next((n for n in author.iter() if _local_name(n.tag) == "LastName"), None)
            collective = next((n for n in author.iter() if _local_name(n.tag) == "CollectiveName"), None)
            if collective is not None and _clean_node_text(collective):
                author_names.append(_clean_node_text(collective))
                continue
            name = " ".join(
                x for x in [_clean_node_text(fore), _clean_node_text(last)] if x
            )
            if name:
                author_names.append(name)

    meta = {
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "journal": journal,
        "publication_year": _extract_pubmed_year(article),
        "authors": ", ".join(author_names),
        "abstract": abstract,
    }
    return title, full_text, meta


def _extract_text_from_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()

    # PubMed EFetch XML and PMC/JATS XML have different schemas.
    if _find_first_local(root, "PubmedArticle") is not None:
        return _extract_pubmed_xml(root, path)
    return _extract_jats_xml(root, path)


def parse_document(path: str):
    if path.lower().endswith(".xml"):
        title, text, meta = _extract_text_from_xml(path)
        root = ET.parse(path).getroot()
        arxiv = root.find(".//ArticleId[@IdType='arxiv']")
        meta['arxiv_id'] = (arxiv.text or '').strip() if arxiv is not None else ''
        return title, text, meta
    raise ValueError("Only .xml files are supported")


def load_corpus(corpus_dir: str):
    documents = []
    if not os.path.isdir(corpus_dir):
        return documents
    for fname in sorted(os.listdir(corpus_dir)):
        fpath = os.path.join(corpus_dir, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith(".xml"):
            continue
        try:
            title, text, meta = parse_document(fpath)
        except (ET.ParseError, ValueError):
            continue
        if text:
            documents.append((title, text, fname, meta))
    return documents


class DuplicateDocumentError(ValueError):
    def __init__(self, document):
        self.document = document
        super().__init__(f"Article already exists: {document.title}")


def find_duplicate_document(title, meta, fname):
    """Find an existing article using stable metadata before indexing a new copy."""
    pmcid = (meta.get("pmcid") or "").strip()
    arxiv_id = (meta.get('arxiv_id') or '').strip()
    if arxiv_id:
        existing = Document.objects.filter(arxiv_id=arxiv_id).first()
        if existing:
            return existing
    if pmcid:
        existing = Document.objects.filter(pmcid__iexact=pmcid).first()
        if existing:
            return existing

    pmid = (meta.get("pmid") or "").strip()
    if pmid:
        existing = Document.objects.filter(pmid__iexact=pmid).first()
        if existing:
            return existing

    doi = (meta.get("doi") or "").strip()
    if doi:
        existing = Document.objects.filter(doi__iexact=doi).first()
        if existing:
            return existing

    # Different stable IDs can legitimately arrive with the same generic filename.
    if fname and not (pmcid or pmid or doi or arxiv_id):
        existing = Document.objects.filter(source_file__iexact=Path(fname).name).first()
        if existing:
            return existing

    # Fallback for XML files that contain neither PMCID/PMID nor DOI.
    title = (title or "").strip()
    if title and not (pmcid or pmid or doi or arxiv_id):
        candidates = Document.objects.filter(title__iexact=title)
        year = (meta.get("publication_year") or "").strip()
        if year:
            candidates = candidates.filter(publication_year=year)
        existing = candidates.first()
        if existing:
            return existing
    return None


def inspect_document(path: str):
    """Parse a file and return metadata plus any matching existing document."""
    title, text, meta = parse_document(path)
    fname = Path(path).name
    return title, text, meta, find_duplicate_document(title, meta, fname)


def _index_one(title, text, fname, meta):
    # Search and statistics deliberately have different scopes. ``text`` is
    # Abstract + Keywords + any supplied PMC body, while the persisted display
    # statistics cover Abstract + Keywords only.
    # Current collection stage searches the extracted Abstract + Keywords only.
    # Keep the complete PMC body in raw_text for future homework stages.
    search_text = meta.get("abstract") or ""
    stats = document_stats(meta.get("abstract") or "")
    doc = Document.objects.create(
        title=title[:500],
        source_file=fname,
        raw_text=text,
        char_count=stats["char_count"],
        word_count=stats["word_count"],
        sentence_count=stats["sentence_count"],
        avg_words_per_sentence=stats["avg_words_per_sentence"],
        pmcid=(meta.get("pmcid") or "")[:40],
        pmid=(meta.get("pmid") or "")[:40],
        arxiv_id=(meta.get('arxiv_id') or '')[:50],
        doi=(meta.get("doi") or "")[:200],
        journal=(meta.get("journal") or "")[:300],
        publication_year=(meta.get("publication_year") or "")[:10],
        authors=meta.get("authors") or "",
        abstract=meta.get("abstract") or "",
    )

    # UI stays simple; preprocessing and the inverted index happen silently here.
    write_postings(doc, search_text)
    return doc


def write_postings(doc, text):
    clear_variant_cache()
    freq = Counter(word[:100] for word in preprocess(text))
    Term.objects.bulk_create([Term(word=word) for word in freq], ignore_conflicts=True)
    terms = Term.objects.filter(word__in=freq).in_bulk(field_name="word")
    Posting.objects.bulk_create(
        [Posting(term=terms[word], document=doc, term_freq=tf) for word, tf in freq.items()],
        batch_size=1000,
    )


@transaction.atomic
def build_index(corpus_dir: str, reset: bool = True):
    # Rebuild postings while keeping document IDs and their topic memberships.
    if reset:
        Posting.objects.all().delete()
        Term.objects.all().delete()

    raw_documents = load_corpus(corpus_dir)
    indexed_ids = set()
    for title, text, fname, meta in raw_documents:
        duplicate = find_duplicate_document(title, meta, fname)
        if duplicate:
            if duplicate.pk not in indexed_ids and (reset or not duplicate.postings.exists()):
                write_postings(duplicate, duplicate.abstract)
            indexed_ids.add(duplicate.pk)
        else:
            indexed_ids.add(_index_one(title, text, fname, meta).pk)
    if reset:
        for doc in Document.objects.filter(postings__isnull=True).distinct():
            write_postings(doc, doc.abstract)
    return Document.objects.count(), Term.objects.count()


@transaction.atomic
def index_file(path: str, replace=False):
    title, text, meta = parse_document(path)
    fname = Path(path).name
    duplicate = find_duplicate_document(title, meta, fname)
    if duplicate and not replace:
        raise DuplicateDocumentError(duplicate)
    if duplicate and replace:
        topic_ids = list(duplicate.topics.values_list("pk", flat=True))
        duplicate.delete()
    doc = _index_one(title, text, fname, meta)
    if duplicate and replace:
        doc.topics.set(topic_ids)
    Term.objects.filter(postings__isnull=True).delete()
    return doc
