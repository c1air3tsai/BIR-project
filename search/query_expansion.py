"""Conservative query-time expansion for punctuation variants."""
import unicodedata

from django.core.cache import cache

from .models import Term
from .text_processing import WORD_CONNECTORS


_VARIANT_CACHE_KEY = "search-term-variants-v1"
_CONNECTORS = set(WORD_CONNECTORS)


def variant_key(term):
    """Return a punctuation-insensitive key for a token, when it is safe.

    Only intra-word hyphen variants are folded. Apostrophes and dots retain
    their meaning, and pure numbers are excluded so numeric ranges do not
    accidentally become equivalent.
    """
    normalized = unicodedata.normalize("NFKC", term or "").casefold()
    if len(normalized) < 3 or not any(char.isalpha() for char in normalized):
        return None
    if any(not (char.isalnum() or char in _CONNECTORS) for char in normalized):
        return None
    compact = "".join(char for char in normalized if char not in _CONNECTORS)
    return compact if len(compact) >= 3 else None


def clear_variant_cache():
    cache.delete(_VARIANT_CACHE_KEY)


def _variant_map():
    mapping = cache.get(_VARIANT_CACHE_KEY)
    if mapping is not None:
        return mapping

    mapping = {}
    for word in Term.objects.order_by("word").values_list("word", flat=True):
        key = variant_key(word)
        if key:
            mapping.setdefault(key, []).append(word)
    cache.set(_VARIANT_CACHE_KEY, mapping, 300)
    return mapping


def expand_index_terms(terms):
    """Group each query term with indexed hyphen/no-hyphen equivalents."""
    mapping = _variant_map()
    groups = {}
    for term in terms:
        key = variant_key(term)
        group_key = ("variant", key) if key else ("exact", term)
        group = groups.setdefault(group_key, {"originals": [], "variants": set()})
        group["originals"].append(term)
        group["variants"].add(term)
        if key:
            group["variants"].update(mapping.get(key, ()))
    return [
        (tuple(group["originals"]), tuple(sorted(group["variants"])))
        for group in groups.values()
    ]
