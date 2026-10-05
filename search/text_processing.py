import math
import re
from functools import lru_cache
from nltk.stem.porter import PorterStemmer

_stemmer = PorterStemmer()

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from", "up", "down",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "this", "that", "these", "those", "it", "its", "as",
    "we", "our", "their", "which", "who", "whom", "not", "no", "can", "will", "would",
    "should", "could", "may", "might", "must", "shall", "than", "so", "such"
}

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "al", "fig", "figs",
    "eq", "eqs", "approx", "no", "nos", "vol", "vols", "pp", "p", "i.e", "e.g",
    "cf", "et", "inc", "ltd", "co", "st"
}

MULTI_DOT_ABBREVIATIONS = {
    "e.g.", "i.e.", "a.m.", "p.m.",
}

# Hyphen-like characters which may join a word when both neighbouring
# characters are letters or numbers.  An em dash is deliberately absent: it
# is always a separator under the project rules.
WORD_CONNECTORS = "-‐‑–"
_NUMERIC_RANGE_BOUNDARY = re.compile(
    rf"(?<=\d)[{re.escape(WORD_CONNECTORS)}](?=\d)"
)

# Shared token rule used by indexing, ranking, statistics and highlighting.
# Examples:
#   COVID-19          -> 1 token
#   SARS-CoV-2        -> 1 token
#   Brca1ΔC/ΔC       -> 2 tokens (slash is a separator)
#   c.1813dupA        -> 2 tokens (period is a separator here)
#   53BP1             -> 1 token
#   12-15%            -> 2 tokens (numeric range)
#   48.1% / 0.05      -> 1 token each (decimal point stays inside number)
#   O'Malley          -> 1 token
#   e.g. / i.e.       -> 1 token each
TOKEN_PATTERN = re.compile(
    # e.g. / i.e. / U.S. -> one token
    r"(?:[A-Za-z]\.){2,}"

    # Numbers / decimals / percentages.
    # 12-15% -> 12 + 15%
    # But 3-y / 5-stage will NOT be split here; the word alternative below
    # consumes those as a single token.
    rf"|\d+(?:\.\d+)*%?(?![\w’']|[{re.escape(WORD_CONNECTORS)}][^\W\d_])"

    # Unicode-aware alphanumeric words.
    # Keep apostrophes and the approved hyphen variants inside words.
    rf"|[^\W_]+(?:['’{re.escape(WORD_CONNECTORS)}][^\W_]+)*",

    flags=re.UNICODE,
)


def _token_scan_text(text: str):
    """Split numeric ranges without changing offsets in the original text."""
    return _NUMERIC_RANGE_BOUNDARY.sub(" ", text or "")


def iter_token_matches(text: str):
    """Yield regex matches using the same boundaries as tokenize()."""
    return TOKEN_PATTERN.finditer(_token_scan_text(text))


def tokenize(text: str):
    """Tokenize biomedical text without stop-word removal or stemming."""
    original = text or ""
    return [
        original[m.start():m.end()].casefold()
        for m in TOKEN_PATTERN.finditer(_token_scan_text(original))
    ]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


@lru_cache(maxsize=50000)
def _stem_token(token):
    return _stemmer.stem(token)


def stem_tokens(tokens):
    return [_stem_token(t) for t in tokens]


def preprocess(text: str):
    """Main IR preprocessing: tokenize -> stop-word removal -> Porter stemming."""
    return stem_tokens(remove_stopwords(tokenize(text)))


def _split_paragraph_sentences(text: str):
    """Apply EOS rules inside one paragraph."""
    sentences = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch not in ".!?。！？":
            i += 1
            continue

        if ch == ".":
            # Decimal number: 3.14 / 0.05 / 48.1
            if (
                0 < i < n - 1
                and text[i - 1].isdigit()
                and text[i + 1].isdigit()
            ):
                i += 1
                continue

            # Multi-dot abbreviations: e.g. / i.e. / a.m. / p.m.
            left = text[max(0, i - 12): i + 1].lower()
            if any(left.endswith(abbr) for abbr in MULTI_DOT_ABBREVIATIONS):
                i += 1
                continue

            # Normal abbreviations: Dr. / Fig. / etc.
            prefix = text[max(start, i - 30): i + 1]
            match = re.search(r"([A-Za-z]+)\.$", prefix)
            prev_word = match.group(1) if match else ""

            if prev_word.lower() in ABBREVIATIONS:
                i += 1
                continue

            # Initial followed by a capitalized name: J. Smith
            j = i + 1
            while j < n and text[j] in "\"'”’)]}":
                j += 1
            while j < n and text[j].isspace():
                j += 1

            if (
                len(prev_word) == 1
                and prev_word.isupper()
                and j < n
                and text[j].isupper()
            ):
                i += 1
                continue

        # Include closing quotation/bracket in the current sentence.
        end = i + 1
        while end < n and text[end] in "\"'”’)]}":
            end += 1

        # A punctuation candidate is an EOS when it is followed by whitespace
        # or the end of the paragraph. We deliberately do not require the next
        # sentence to start with an uppercase letter because biomedical text can
        # start with gene names, symbols or lowercase technical terms.
        if end >= n or text[end].isspace():
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)

            start = end
            while start < n and text[start].isspace():
                start += 1
            i = start
            continue

        i += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str):
    """
    Rule-based sentence segmentation.

    Paragraph boundaries are handled separately, then EOS rules are applied
    within each paragraph. Guards cover decimals, common abbreviations,
    initials and closing quotes/brackets.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\r?\n+", text) if p.strip()]

    sentences = []
    for paragraph in paragraphs:
        sentences.extend(_split_paragraph_sentences(paragraph))
    return sentences


def document_stats(text: str):
    raw_text = text or ""
    visible_text = re.sub(r"\s+", " ", raw_text).strip()
    word_count = len(tokenize(visible_text))
    sentence_count = len(split_sentences(raw_text))
    return {
        "char_count": len(visible_text),
        "char_count_no_space": len(visible_text.replace(" ", "")),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(word_count / sentence_count, 2) if sentence_count else 0,
    }


def bm25_score(tf, df, doc_len, avg_doc_len, total_docs, k1=1.2, b=0.75):
    """Classic BM25 contribution for one query term/document pair."""
    if tf <= 0 or df <= 0 or total_docs <= 0:
        return 0.0
    avg_doc_len = avg_doc_len or 1
    idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
    denom = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
    return idf * ((tf * (k1 + 1)) / denom)
