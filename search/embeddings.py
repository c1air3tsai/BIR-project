"""Reproducible, cached Word2Vec experiments over the shared Zipf tokenizer."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re

from django.conf import settings

from .analysis import CONDITIONS, condition_tokens
from .text_processing import STOPWORDS, split_sentences

SEED = 42
NEGATIVE = 5
SUBSAMPLING = 0.001
WORKERS = 1
CACHE_VERSION = 6
PREPROCESSING_VERSION = "condition-pipeline-v2"
DEFAULTS = {
    "condition": "C",
    "dimensions": 100,
    "window": 5,
    "min_count": 2,
    "epochs": 20,
}
EXPERIMENTS = {
    "window": ("Context window", 2, 5),
    "dimensions": ("Embedding dimension", 50, 100),
    "min_count": ("Minimum count", 2, 5),
}
EXPERIMENT_LIMITS = {
    "window": (1, 20),
    "dimensions": (10, 500),
    "min_count": (1, 50),
}
PRESET_SPECS = {
    "medical": {"slug": "glp1_1000", "label": "GLP-1 only", "medical": 1000, "cs": 0},
    "balanced": {"slug": "balanced_500_500", "label": "Balanced", "medical": 500, "cs": 500},
    "cs": {"slug": "cs_500", "label": "CS only", "medical": 0, "cs": 500},
}
logger = logging.getLogger(__name__)


def stable_hash(value):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def parameters(values):
    out = dict(DEFAULTS)
    choices_by_name = {
        "condition": list(CONDITIONS),
        "epochs": [10, 20, 40],
    }
    for name, choices in choices_by_name.items():
        value = values.get(name, out[name])
        if isinstance(out[name], int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                value = out[name]
        out[name] = value if value in choices else out[name]
    for name, (low, high) in EXPERIMENT_LIMITS.items():
        try:
            value = int(values.get(name, out[name]))
        except (TypeError, ValueError):
            value = out[name]
        out[name] = value if low <= value <= high else out[name]
    return out


def full_parameters(params):
    """Every setting that affects training and therefore belongs in the key."""
    return {
        **parameters(params),
        "negative": NEGATIVE,
        "sample": SUBSAMPLING,
        "seed": SEED,
        "workers": WORKERS,
    }


def _architecture_parameters(params, architecture):
    return {**full_parameters(params), "architecture": architecture, "sg": int(architecture == "skipgram")}


def cache_root():
    return Path(getattr(settings, "EMBEDDING_CACHE_DIR", Path(settings.BASE_DIR) / "data" / "embedding_cache"))


def _record_digests(records):
    identifiers = hashlib.sha256()
    content = hashlib.sha256()
    normalized = sorted((_record_parts(record) for record in records), key=lambda row: (row[0], row[2]))
    for pk, text, domain in normalized:
        identifiers.update(f"{domain}\0{pk}\0".encode("utf-8"))
        content.update(f"{domain}\0{pk}\0{text}\0".encode("utf-8"))
    return identifiers.hexdigest(), content.hexdigest()


def shared_cache_key(records, params):
    document_ids_hash, content_hash = _record_digests(records)
    payload = {
        "cache_version": CACHE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "document_ids_hash": document_ids_hash,
        "content_hash": content_hash,
        "domain_counts": dict(sorted(Counter(_record_parts(record)[2] for record in records).items())),
        "settings": full_parameters(params),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def preset_cache_name(selection, params):
    spec = PRESET_SPECS.get(selection.get("preset"))
    if not spec or any(parameters(params)[key] != DEFAULTS[key] for key in DEFAULTS):
        return None
    expected = {
        "medical": min(spec["medical"], selection["available"]["medical"]),
        "cs": min(spec["cs"], selection["available"]["cs"]),
    }
    return spec["slug"] if selection["actual"] == expected and sum(expected.values()) > 0 else None


def cache_directory(records, params, cache_name=None):
    return cache_root() / (cache_name or "custom") / ("" if cache_name else shared_cache_key(records, params))


def _manifest_expected(records, params, preset_name=None):
    document_ids_hash, content_hash = _record_digests(records)
    counts = Counter(_record_parts(record)[2] for record in records)
    return {
        "cache_version": CACHE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "preset_name": preset_name or "custom",
        "document_count": len(records),
        "document_ids": [_record_parts(record)[0] for record in sorted(records, key=lambda row: _record_parts(row)[0])],
        "document_ids_hash": document_ids_hash,
        "content_hash": content_hash,
        "domain_counts": {"medical": counts["medical"], "cs": counts["cs"]},
        "model_settings": full_parameters(params),
        "shared_cache_key": shared_cache_key(records, params),
    }


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_manifest(records, params, cache_name=None):
    directory = cache_directory(records, params, cache_name)
    manifest = _read_json(directory / "manifest.json")
    expected = _manifest_expected(records, params, cache_name)
    if not manifest or any(manifest.get(key) != value for key, value in expected.items()):
        return None
    return manifest


def _update_manifest(records, params, cache_name=None, **updates):
    directory = cache_directory(records, params, cache_name)
    manifest = load_manifest(records, params, cache_name) or {
        **_manifest_expected(records, params, cache_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "gensim": None,
            "scikit_learn": None,
        },
        "models": {},
        "artifacts": {},
        "timings": {},
    }
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key].update(value)
        else:
            manifest[key] = value
    _write_json(directory / "manifest.json", manifest)
    return manifest


def corpus_records(documents):
    """Return every eligible abstract; Word Embedding has no hidden hard cap."""
    rows = documents.exclude(abstract="").order_by("pk").values_list("pk", "abstract")
    return [(pk, text) for pk, text in rows if text and text.strip()]


def _record_parts(record):
    return record[0], record[1], record[2] if len(record) > 2 else "unknown"


def _sample_key(record, domain):
    pk, _text, _kind = _record_parts(record)
    return hashlib.sha256(f"{SEED}:{domain}:{pk}".encode()).digest()


def select_corpus(documents, values):
    """Select one shared, deterministic Medical/CS corpus for both architectures."""
    raw = [
        (pk, abstract, "cs" if arxiv_id else "medical")
        for pk, abstract, arxiv_id in documents.exclude(abstract="").order_by("pk").values_list("pk", "abstract", "arxiv_id")
        if abstract and abstract.strip()
    ]
    pools = {
        "medical": sorted((row for row in raw if row[2] == "medical"), key=lambda row: _sample_key(row, "medical")),
        "cs": sorted((row for row in raw if row[2] == "cs"), key=lambda row: _sample_key(row, "cs")),
    }
    available = {key: len(rows) for key, rows in pools.items()}
    preset = values.get("corpus_preset", "medical")
    if preset not in {"medical", "cs", "balanced", "all", "custom"}:
        preset = "balanced"
    balanced = min(500, available["medical"], available["cs"])
    presets = {
        "medical": (min(1000, available["medical"]), 0), "cs": (0, min(500, available["cs"])),
        "balanced": (balanced, balanced), "all": (available["medical"], available["cs"]),
    }
    errors = []
    if preset == "custom":
        requested = []
        for key in ("medical_count", "cs_count"):
            try:
                requested.append(int(values.get(key, balanced)))
            except (TypeError, ValueError):
                requested.append(-1)
        medical_count, cs_count = requested
    else:
        medical_count, cs_count = presets[preset]
    if not 0 <= medical_count <= available["medical"]:
        errors.append("medical_count")
    if not 0 <= cs_count <= available["cs"]:
        errors.append("cs_count")
    if medical_count <= 0 and cs_count <= 0:
        errors.append("empty")
    medical_count = min(max(medical_count, 0), available["medical"])
    cs_count = min(max(cs_count, 0), available["cs"])
    records = pools["medical"][:medical_count] + pools["cs"][:cs_count]
    records.sort(key=lambda row: row[0])
    digest = hashlib.sha256()
    for pk, text, domain in records:
        digest.update(f"{domain}\0{pk}\0{text}\0".encode("utf-8"))
    content_hash = digest.hexdigest()
    return {
        "preset": preset, "preset_label": PRESET_SPECS.get(preset, {}).get("label", "Custom" if preset == "custom" else "All Available"), "available": available,
        "requested": {"medical": medical_count, "cs": cs_count},
        "actual": {"medical": sum(row[2] == "medical" for row in records), "cs": sum(row[2] == "cs" for row in records)},
        "records": records, "document_ids": [row[0] for row in records],
        "content_hash": content_hash, "run_id": content_hash[:12], "seed": SEED,
        "valid": not errors, "errors": errors,
    }


def model_key(records, params):
    architecture = params.get("architecture", "skipgram")
    payload = {"shared": shared_cache_key(records, params), "architecture": architecture}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def model_path(records, params, cache_name=None):
    architecture = params.get("architecture", "skipgram")
    return cache_directory(records, params, cache_name) / f"{architecture}.model"


def sentences_for(records, condition):
    return [
        tokens
        for record in records
        for text in [_record_parts(record)[1]]
        for sentence in split_sentences(text)
        if len(tokens := condition_tokens(sentence)[condition]) >= 2
    ]


def corpus_statistics(records, condition):
    sentences = sentences_for(records, condition)
    counts = Counter(token for sentence in sentences for token in sentence)
    return {
        "documents": len(records),
        "sentences": len(sentences),
        "input_tokens": sum(counts.values()),
        "unfiltered_vocabulary": len(counts),
        "top_terms": [
            {"term": term, "cf": count} for term, count in counts.most_common(15)
        ],
        "sentences_data": sentences,
        "frequencies": counts,
    }


class _EpochLossLogger:
    """Gensim callback that converts cumulative loss into real epoch deltas."""

    def __init__(self):
        self.rows = []
        self._previous = 0.0
        self._started = None

    def on_train_begin(self, model):
        self._previous = 0.0

    def on_train_end(self, model):
        return None

    def on_epoch_begin(self, model):
        self._started = time.monotonic()

    def on_epoch_end(self, model):
        cumulative = float(model.get_latest_training_loss())
        delta = cumulative - self._previous if cumulative >= self._previous else cumulative
        self._previous = cumulative
        self.rows.append(
            {
                "epoch": len(self.rows) + 1,
                "loss": max(0.0, float(delta)),
                "cumulative_loss": cumulative,
                "seconds": round(time.monotonic() - self._started, 4),
            }
        )


def training_sample(sentences, window):
    if not sentences:
        return {}
    sentence = max(sentences[: min(len(sentences), 200)], key=len)
    center_index = len(sentence) // 2
    center = sentence[center_index]
    left = max(0, center_index - window)
    right = min(len(sentence), center_index + window + 1)
    context = [sentence[i] for i in range(left, right) if i != center_index]
    return {
        "sentence": " ".join(sentence),
        "window": window,
        "center": center,
        "context": context,
        "skipgram": [{"input": center, "target": word} for word in context],
        "cbow": {"input": context, "target": center},
    }


def train(records, params, force=False, cache_name=None):
    """Train or reuse a model with measured per-epoch loss and timing."""
    import gensim
    from gensim.models import Word2Vec

    source_params = dict(params)
    architecture = source_params.get("architecture", "skipgram")
    params = parameters(source_params)
    params["architecture"] = architecture
    destination = model_path(records, params, cache_name)
    manifest = load_manifest(records, params, cache_name)
    if destination.exists() and manifest and architecture in manifest.get("models", {}) and not force:
        return Word2Vec.load(str(destination))

    preprocessing_started = time.monotonic()
    stats = corpus_statistics(records, params["condition"])
    sentences = stats.pop("sentences_data")
    stats.pop("frequencies")
    preprocessing_seconds = round(time.monotonic() - preprocessing_started, 4)
    if not sentences:
        raise ValueError("No sentences with at least two tokens. Import more abstracts first.")

    started = time.monotonic()
    model = Word2Vec(
        vector_size=params["dimensions"],
        window=params["window"],
        min_count=params["min_count"],
        sg=int(params["architecture"] == "skipgram"),
        workers=WORKERS,
        seed=SEED,
        hashfxn=stable_hash,
        negative=NEGATIVE,
        hs=0,
        sample=SUBSAMPLING,
        epochs=params["epochs"],
    )
    model.build_vocab(sentences)
    if len(model.wv) < 2:
        raise ValueError("Fewer than two eligible words. Lower minimum count or import more abstracts.")
    loss_logger = _EpochLossLogger()
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=params["epochs"],
        compute_loss=True,
        callbacks=[loss_logger],
    )
    elapsed = round(time.monotonic() - started, 3)
    losses = loss_logger.rows
    domain_counts = Counter(_record_parts(record)[2] for record in records)
    corpus_digest = hashlib.sha256()
    for record in records:
        pk, text, domain = _record_parts(record)
        corpus_digest.update(f"{domain}\0{pk}\0{text}\0".encode("utf-8"))
    model.bir_metadata = {
        "parameters": _architecture_parameters(params, architecture),
        **stats,
        "training_vocabulary": len(model.wv),
        "seconds": elapsed,
        "losses": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
        "average_loss": sum(row["loss"] for row in losses) / len(losses) if losses else None,
        "implementation": "gensim Word2Vec",
        "scope": "Selected abstracts only; abstract and sentence boundaries preserved",
        "document_ids": [_record_parts(record)[0] for record in records],
        "domain_counts": {"medical": domain_counts["medical"], "cs": domain_counts["cs"]},
        "corpus_hash": corpus_digest.hexdigest(),
        "preprocessing": CONDITIONS[params["condition"]][1],
        "sample_preview": training_sample(sentences, params["window"]),
        "cache_key": model_key(records, params),
        "cache_version": CACHE_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_seconds": preprocessing_seconds,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".model")
    os.close(fd)
    try:
        model.save(temporary, separately=[])
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    _write_json(destination.parent / f"{architecture}_loss.json", losses)
    _write_json(destination.parent / f"{architecture}_metadata.json", model.bir_metadata)
    _update_manifest(
        records,
        params,
        cache_name,
        software={"gensim": gensim.__version__},
        models={architecture: {
            "file": destination.name,
            "cache_key": model.bir_metadata["cache_key"],
            "training_time_seconds": elapsed,
            "vocabulary_size": len(model.wv),
            "loss_file": f"{architecture}_loss.json",
            "metadata_file": f"{architecture}_metadata.json",
        }},
        timings={"preprocessing_seconds": preprocessing_seconds, f"{architecture}_training_seconds": elapsed},
    )
    logger.info("Word2Vec %s training: %.3f sec; preprocessing: %.4f sec", architecture, elapsed, preprocessing_seconds)
    return model


def load(records, params, cache_name=None):
    source_params = dict(params)
    architecture = source_params.get("architecture", "skipgram")
    normalized = {**parameters(source_params), "architecture": architecture}
    path = model_path(records, normalized, cache_name)
    manifest = load_manifest(records, normalized, cache_name)
    if not path.exists() or not manifest or architecture not in manifest.get("models", {}):
        return None
    from gensim.models import Word2Vec

    return Word2Vec.load(str(path))


def train_both(records, params, force=False, cache_name=None):
    return {
        architecture: train(records, {**parameters(params), "architecture": architecture}, force=force, cache_name=cache_name)
        for architecture in ("cbow", "skipgram")
    }


def load_both(records, params, cache_name=None):
    return {
        architecture: load(records, {**parameters(params), "architecture": architecture}, cache_name=cache_name)
        for architecture in ("cbow", "skipgram")
    }


def normalize_word(text, condition):
    tokens = condition_tokens(text)[condition]
    if len(tokens) != 1:
        raise ValueError("Enter one word or biomedical token, such as learning or GLP-1.")
    return tokens[0]


def neighbors(model, text, condition, count=10):
    word = normalize_word(text, condition)
    if word not in model.wv:
        raise ValueError(
            f"“{word}” is outside this model vocabulary. Try a frequent term or lower minimum count."
        )
    rows = [
        {"term": term, "similarity": float(score)}
        for term, score in model.wv.most_similar(word, topn=min(count, len(model.wv) - 1))
    ]
    return word, rows


def cosine(model, word_a, word_b):
    """Cosine similarity in the complete Word2Vec vector space."""
    import numpy as np

    a = np.asarray(model.wv[word_a], dtype=float)
    b = np.asarray(model.wv[word_b], dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def normalized_vectors(model, words):
    import numpy as np

    matrix = np.array([model.wv[word] for word in words], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _pca_points(model, words, normalized=True):
    import numpy as np
    from sklearn.decomposition import PCA

    matrix = normalized_vectors(model, words) if normalized else np.array([model.wv[word] for word in words], dtype=float)
    dimensions = min(2, len(words), matrix.shape[1])
    fitted = PCA(n_components=dimensions, svd_solver="full").fit(matrix)
    coordinates = fitted.transform(matrix)
    if dimensions == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(words))))
    ratios = [float(value) for value in fitted.explained_variance_ratio_]
    while len(ratios) < 2:
        ratios.append(0.0)
    return coordinates, ratios


def projection(model, words):
    coordinates, ratios = _pca_points(model, words, normalized=True)
    return {
        "method": "PCA of L2-normalized Word2Vec vectors",
        "points": [
            {"term": word, "x": float(point[0]), "y": float(point[1])}
            for word, point in zip(words, coordinates)
        ],
        "variance": sum(ratios),
        "variance_ratio": ratios,
    }


def vocabulary_projection(model, count=40):
    words = model.wv.index_to_key[: min(count, len(model.wv))]
    result = projection(model, words)
    for rank, point in enumerate(result["points"], 1):
        point["rank"] = rank
        point["count"] = int(model.wv.get_vecattr(point["term"], "count"))
    return result


def _visualization_cache_path(model, words, selected=""):
    key = hashlib.sha256(
        json.dumps(
            {"version": 3, "model": model.bir_metadata.get("cache_key"), "words": words, "selected": selected},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return cache_root() / "visualizations" / (key + ".json")


def semantic_visualizations(model, words, selected="", destination=None, force=False):
    """Cached normalized-vector PCA/t-SNE plus neighborhood preservation diagnostics."""
    words = [word for word in dict.fromkeys(words) if word in model.wv]
    if len(words) < 2:
        return {"pca": {"points": []}, "tsne": {"points": []}}
    destination = Path(destination) if destination else _visualization_cache_path(model, words, selected)
    if destination.exists() and not force:
        cached = _read_json(destination)
        if cached:
            return cached

    import numpy as np
    import sklearn
    from sklearn.manifold import TSNE, trustworthiness

    pca_started = time.monotonic()
    pca = projection(model, words)
    pca_seconds = round(time.monotonic() - pca_started, 4)
    matrix = normalized_vectors(model, words)
    perplexity = min(30, max(1, min(len(words) - 1, (len(words) - 1) // 3 or 1)))
    tsne_started = time.monotonic()
    coordinates = TSNE(
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        init="pca" if len(words) > 2 else "random",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(matrix)
    tsne_seconds = round(time.monotonic() - tsne_started, 4)
    tsne = {
        "method": "t-SNE of L2-normalized Word2Vec vectors (Euclidean metric)",
        "perplexity": perplexity,
        "points": [
            {"term": word, "x": float(point[0]), "y": float(point[1])}
            for word, point in zip(words, coordinates)
        ],
    }
    counts = {word: int(model.wv.get_vecattr(word, "count")) for word in words}
    for chart in (pca, tsne):
        for rank, point in enumerate(chart["points"], 1):
            point.update(rank=rank, count=counts[point["term"]])
        chart["selected"] = selected
        chart["number_of_points"] = len(words)
        chart["seed"] = SEED
    neighbor_count = min(10, max(1, (len(words) - 1) // 2))
    trustworthiness_started = time.monotonic()
    if len(words) > neighbor_count + 1:
        pca["trustworthiness"] = float(trustworthiness(matrix, [[p["x"], p["y"]] for p in pca["points"]], n_neighbors=neighbor_count))
        tsne["trustworthiness"] = float(trustworthiness(matrix, coordinates, n_neighbors=neighbor_count))
    else:
        pca["trustworthiness"] = tsne["trustworthiness"] = None
    trustworthiness_seconds = round(time.monotonic() - trustworthiness_started, 4)
    pca["n_neighbors"] = tsne["n_neighbors"] = neighbor_count
    result = {
        "pca": pca,
        "tsne": tsne,
        "timings": {
            "pca_seconds": pca_seconds,
            "tsne_seconds": tsne_seconds,
            "trustworthiness_seconds": trustworthiness_seconds,
        },
        "metadata": {
            "word_count": len(words),
            "word_set_hash": hashlib.sha256("\0".join(words).encode("utf-8")).hexdigest(),
            "normalization": "temporary L2; model vectors unchanged",
            "scikit_learn": sklearn.__version__,
        },
    }
    _write_json(destination, result)
    if destination.name.endswith("_global.json"):
        prefix = destination.name.removesuffix("_global.json")
        _write_json(destination.with_name(f"{prefix}_global_pca.json"), pca)
        _write_json(destination.with_name(f"{prefix}_global_tsne.json"), tsne)
    logger.info("%s PCA: %.4f sec; t-SNE: %.4f sec; trustworthiness: %.4f sec", destination.name, pca_seconds, tsne_seconds, trustworthiness_seconds)
    return result


def default_targets(models, count=10):
    available = set(models["cbow"].wv.index_to_key) & set(models["skipgram"].wv.index_to_key)
    return [word for word in models["cbow"].wv.index_to_key if word in available][:count]


def _valid_content_term(word):
    return len(word) >= 2 and any(char.isalpha() for char in word) and not word.isnumeric()


def global_words(models, count=200):
    """One deterministic content-word set shared by both architectures and projections."""
    common = set(models["cbow"].wv.index_to_key) & set(models["skipgram"].wv.index_to_key)
    scored = []
    for word in common:
        if not _valid_content_term(word) or word in STOPWORDS:
            continue
        cf = min(int(models["cbow"].wv.get_vecattr(word, "count")), int(models["skipgram"].wv.get_vecattr(word, "count")))
        if cf >= 2:
            scored.append((word, cf))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [word for word, _ in scored[:count]]


def parse_targets(raw, models, condition, count=10):
    """Return defaults only for an empty query; otherwise keep valid requested terms."""
    requested = []
    for value in (raw or "").replace("\n", ",").split(","):
        value = value.strip()
        if not value:
            continue
        tokens = condition_tokens(value)[condition]
        if len(tokens) == 1:
            requested.append(tokens[0])
    common = set(models["cbow"].wv.index_to_key) & set(models["skipgram"].wv.index_to_key)
    selected = [word for word in dict.fromkeys(requested) if word in common]
    if (raw or "").strip():
        return selected[:count]
    return default_targets(models, count)


def neighbor_comparison(models, targets, topn=5):
    rows = []
    for target in targets:
        by_architecture = {}
        for architecture, model in models.items():
            by_architecture[architecture] = [
                {"rank": rank, "term": term, "similarity": float(score)}
                for rank, (term, score) in enumerate(model.wv.most_similar(target, topn=topn), 1)
            ]
        left = {row["term"] for row in by_architecture["cbow"]}
        right = {row["term"] for row in by_architecture["skipgram"]}
        rows.append(
            {
                "target": target,
                **by_architecture,
                "overlap": sorted(left & right),
                "jaccard": len(left & right) / len(left | right) if left | right else 0.0,
            }
        )
    return rows


def semantic_pairs(models, count=5):
    common = set(models["cbow"].wv.index_to_key) & set(models["skipgram"].wv.index_to_key)
    preferred = [
        ("cancer", "tumor"), ("treatment", "therapy"),
        ("diagnosis", "disease"), ("glucose", "insulin"),
        ("computer", "algorithm"), ("learning", "model"),
        ("patient", "clinical"), ("retrieval", "document"),
    ]
    pairs = [pair for pair in preferred if pair[0] in common and pair[1] in common]
    used = {tuple(sorted(pair)) for pair in pairs}
    for word in models["cbow"].wv.index_to_key:
        if len(pairs) >= count or word not in common:
            continue
        for neighbor, _ in models["cbow"].wv.most_similar(word, topn=min(10, len(common) - 1)):
            key = tuple(sorted((word, neighbor)))
            if neighbor in common and neighbor != word and key not in used:
                pairs.append((word, neighbor))
                used.add(key)
                break
    output = []
    for word_a, word_b in pairs[:count]:
        output.append(
            {
                "word_a": word_a,
                "word_b": word_b,
                "cbow": cosine(models["cbow"], word_a, word_b),
                "skipgram": cosine(models["skipgram"], word_a, word_b),
                "cbow_neighbors": [term for term, _ in models["cbow"].wv.most_similar(word_a, topn=5)],
                "skipgram_neighbors": [term for term, _ in models["skipgram"].wv.most_similar(word_a, topn=5)],
            }
        )
    return output


def rare_word_analysis(models, frequencies, count=5):
    common = set(models["cbow"].wv.index_to_key) & set(models["skipgram"].wv.index_to_key)
    candidates = sorted(
        (word for word in common if 2 <= frequencies.get(word, 0) <= 10 and _valid_content_term(word)),
        key=lambda word: (frequencies[word], word),
    )
    if len(candidates) < count:
        candidates = sorted(
            (word for word in common if _valid_content_term(word)),
            key=lambda word: (frequencies.get(word, 0), word),
        )
    output = []
    for word in candidates[:count]:
        output.append(
            {
                "word": word,
                "frequency": frequencies.get(word, 0),
                "cbow": [{"term": term, "similarity": float(score)} for term, score in models["cbow"].wv.most_similar(word, topn=5)],
                "skipgram": [{"term": term, "similarity": float(score)} for term, score in models["skipgram"].wv.most_similar(word, topn=5)],
            }
        )
    return output


def build_global_artifacts(records, params, models, cache_name=None, force=False):
    """Build every model-dependent artifact once and persist it beside the pair."""
    directory = cache_directory(records, params, cache_name)
    artifact_started = time.monotonic()
    stats = corpus_statistics(records, parameters(params)["condition"])
    frequencies = stats.pop("frequencies")
    stats.pop("sentences_data")
    words = global_words(models, 200)
    targets = default_targets(models, 10)
    semantic = {
        "targets": targets,
        "neighbors": neighbor_comparison(models, targets),
        "pairs": semantic_pairs(models),
    }
    rare = rare_word_analysis(models, frequencies)
    visualizations = {}
    timings = {}
    for architecture, model in models.items():
        visual = semantic_visualizations(
            model,
            words,
            destination=directory / f"{architecture}_global.json",
            force=force,
        )
        visualizations[architecture] = visual
        timings.update({f"{architecture}_{key}": value for key, value in visual.get("timings", {}).items()})
    _write_json(directory / "corpus_statistics.json", stats)
    _write_json(directory / "global_words.json", words)
    _write_json(directory / "semantic_metadata.json", semantic)
    _write_json(directory / "rare_word_metadata.json", rare)
    total_seconds = round(time.monotonic() - artifact_started, 4)
    timings["artifact_total_seconds"] = total_seconds
    _update_manifest(
        records,
        params,
        cache_name,
        software={"scikit_learn": visualizations["cbow"].get("metadata", {}).get("scikit_learn")},
        artifacts={
            "corpus_statistics": "corpus_statistics.json",
            "global_words": "global_words.json",
            "semantic_metadata": "semantic_metadata.json",
            "rare_word_metadata": "rare_word_metadata.json",
            "cbow_global": "cbow_global.json",
            "skipgram_global": "skipgram_global.json",
        },
        timings=timings,
    )
    return {
        "statistics": stats,
        "words": words,
        "semantic": semantic,
        "rare": rare,
        "visualizations": visualizations,
        "timings": timings,
    }


def load_global_artifacts(records, params, cache_name=None):
    directory = cache_directory(records, params, cache_name)
    manifest = load_manifest(records, params, cache_name)
    if not manifest:
        return None
    names = {
        "statistics": "corpus_statistics.json",
        "words": "global_words.json",
        "semantic": "semantic_metadata.json",
        "rare": "rare_word_metadata.json",
        "cbow": "cbow_global.json",
        "skipgram": "skipgram_global.json",
    }
    loaded = {key: _read_json(directory / filename) for key, filename in names.items()}
    if any(value is None for value in loaded.values()):
        return None
    return {
        "statistics": loaded["statistics"],
        "words": loaded["words"],
        "semantic": loaded["semantic"],
        "rare": loaded["rare"],
        "visualizations": {"cbow": loaded["cbow"], "skipgram": loaded["skipgram"]},
        "timings": manifest.get("timings", {}),
    }


def cache_status(records, params, cache_name=None, load_seconds=0.0):
    manifest = load_manifest(records, params, cache_name)
    ready = bool(manifest and all((cache_directory(records, params, cache_name) / f"{name}.model").exists() for name in ("cbow", "skipgram")))
    return {
        "ready": ready,
        "state": "pretrained" if ready and cache_name else ("cached" if ready else "missing"),
        "preset": manifest.get("preset_name") if manifest else cache_name,
        "created_at": manifest.get("created_at") if manifest else None,
        "run_id": manifest.get("shared_cache_key", "")[:12] if manifest else shared_cache_key(records, params)[:12],
        "path": str(cache_directory(records, params, cache_name)),
        "load_seconds": round(load_seconds, 4),
        "manifest": manifest,
    }


def precompute_preset(documents, preset, force=False):
    if preset not in PRESET_SPECS:
        raise ValueError(f"Unknown pretrained preset: {preset}")
    selection_started = time.monotonic()
    selection = select_corpus(documents, {"corpus_preset": preset})
    selection_seconds = round(time.monotonic() - selection_started, 4)
    params = dict(DEFAULTS)
    cache_name = PRESET_SPECS[preset]["slug"]
    existing = load_manifest(selection["records"], params, cache_name)
    artifacts = load_global_artifacts(selection["records"], params, cache_name) if existing and not force else None
    models_ready = existing and all((cache_directory(selection["records"], params, cache_name) / f"{name}.model").exists() for name in ("cbow", "skipgram"))
    if models_ready and artifacts and not force:
        return {"preset": preset, "selection": selection, "cache_name": cache_name, "cache_hit": True, "manifest": existing}
    models = train_both(selection["records"], params, force=force, cache_name=cache_name)
    build_global_artifacts(selection["records"], params, models, cache_name=cache_name, force=force)
    manifest = _update_manifest(selection["records"], params, cache_name, timings={"corpus_selection_seconds": selection_seconds})
    logger.info("Preset %s corpus selection: %.4f sec", preset, selection_seconds)
    return {"preset": preset, "selection": selection, "cache_name": cache_name, "cache_hit": False, "manifest": manifest}


def comparison_table(models, rare_rows=None):
    cbow = models["cbow"].bir_metadata
    skipgram = models["skipgram"].bir_metadata
    faster = "CBOW" if cbow["seconds"] < skipgram["seconds"] else "Skip-Gram"
    rare_rows = rare_rows or []
    cbow_rare = "; ".join(f'{row["word"]}→{row["cbow"][0]["term"]}' for row in rare_rows[:3]) or "No shared rare terms"
    skipgram_rare = "; ".join(f'{row["word"]}→{row["skipgram"][0]["term"]}' for row in rare_rows[:3]) or "No shared rare terms"
    return [
        ("Input", "Surrounding context words", "Center word"),
        ("Prediction target", "Center word", "Surrounding context words"),
        ("Training examples", "One context → center example per position", "Multiple center → context pairs per position"),
        ("Training speed", f'{cbow["seconds"]} s', f'{skipgram["seconds"]} s ({faster} was faster here)'),
        ("Final / average training loss", f'{cbow["final_loss"]:.3f} / {cbow["average_loss"]:.3f}', f'{skipgram["final_loss"]:.3f} / {skipgram["average_loss"]:.3f}'),
        ("Vocabulary size", str(cbow["training_vocabulary"]), str(skipgram["training_vocabulary"])),
        ("Rare words", cbow_rare, skipgram_rare),
        ("Expected behavior", "Smooths context evidence when predicting a center", "Creates several prediction signals from each center"),
    ]


def experiment_settings(values):
    """Return validated A/B values for every controlled comparison."""
    settings = {}
    for kind, (_, default_a, default_b) in EXPERIMENTS.items():
        low, high = EXPERIMENT_LIMITS[kind]
        pair = []
        for suffix, default in (("a", default_a), ("b", default_b)):
            try:
                value = int(values.get(f"compare_{kind}_{suffix}", default))
            except (TypeError, ValueError):
                value = default
            pair.append(max(low, min(high, value)))
        if pair[0] == pair[1]:
            pair = [default_a, default_b]
        settings[f"compare_{kind}_a"], settings[f"compare_{kind}_b"] = pair
    return settings


def experiment_config(params, kind):
    if kind not in EXPERIMENTS:
        kind = "window"
    label, _, _ = EXPERIMENTS[kind]
    custom = experiment_settings(params)
    first = custom[f"compare_{kind}_a"]
    second = custom[f"compare_{kind}_b"]
    base = parameters(params)
    return kind, label, ({**base, kind: first}, {**base, kind: second})


def experiment_result_path(records, params, kind):
    kind, _, configs = experiment_config(params, kind)
    digest = hashlib.sha256(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "kind": kind,
                "model_keys": [shared_cache_key(records, config) for config in configs],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return cache_root() / "experiments" / f"{digest}.json"


def load_experiment_result(records, params, kind):
    return _read_json(experiment_result_path(records, params, kind))


def train_experiment(records, params, kind, base_models=None):
    kind, label, configs = experiment_config(params, kind)
    base = parameters(params)
    models = [
        base_models if config == base and base_models and all(base_models.values()) else train_both(records, config)
        for config in configs
    ]
    target_candidates = default_targets(models[0], 1) if all(models[0].values()) else []
    target = target_candidates[0] if target_candidates else ""
    result = experiment_result(label, kind, configs, models, target)
    if result:
        _write_json(experiment_result_path(records, params, kind), result)
    return kind, label, configs, models


def load_experiment(records, params, kind):
    kind, label, configs = experiment_config(params, kind)
    return kind, label, configs, [load_both(records, config) for config in configs]


def experiment_result(label, kind, configs, models, target):
    if not all(all(pair.values()) for pair in models):
        return None
    all_models = [pair[architecture] for pair in models for architecture in ("cbow", "skipgram")]
    common = set(all_models[0].wv.index_to_key)
    for model in all_models[1:]:
        common &= set(model.wv.index_to_key)
    if target not in common:
        target = next((word for word in all_models[0].wv.index_to_key if word in common), "")
    rows = []
    overlaps = {}
    for config, pair in zip(configs, models):
        for architecture in ("cbow", "skipgram"):
            model = pair[architecture]
            nearest = model.wv.most_similar(target, topn=min(5, len(model.wv) - 1)) if target else []
            rows.append({"value": config[kind], "architecture": architecture, "params": full_parameters({**config,"architecture":architecture}),
                "seconds": model.bir_metadata["seconds"], "vocabulary": len(model.wv), "final_loss": model.bir_metadata["final_loss"],
                "neighbors": [{"term": term, "similarity": float(score)} for term, score in nearest]})
    for architecture in ("cbow", "skipgram"):
        matched=[row for row in rows if row["architecture"]==architecture]
        left={item["term"] for item in matched[0]["neighbors"]};right={item["term"] for item in matched[1]["neighbors"]}
        overlaps[architecture]=len(left&right)/len(left|right) if left|right else 0.0
    return {
        "kind": kind, "label": label, "target": target, "rows": rows,
        "overlaps": overlaps,
        "description": f"Only {label} changed; corpus and all other settings were fixed, so observed differences are primarily attributable to {label}.",
    }


def evidence_report(models, neighbor_rows, pair_rows, rare_rows, experiment=None, visualization=None):
    cbow = models["cbow"].bir_metadata
    skipgram = models["skipgram"].bir_metadata
    average_overlap = sum(row["jaccard"] for row in neighbor_rows) / len(neighbor_rows) if neighbor_rows else 0.0
    rare_summary = "; ".join(
        f'{row["word"]} (CF={row["frequency"]}): CBOW {", ".join(row["cbow"][:2])}; Skip-Gram {", ".join(row["skipgram"][:2])}'
        for row in rare_rows
    )
    pair_summary = "; ".join(
        f'{row["word_a"]}/{row["word_b"]}: CBOW {row["cbow"]:.3f}, Skip-Gram {row["skipgram"]:.3f}'
        for row in pair_rows
    )
    rq4 = (
        f'{experiment["description"]} For “{experiment["target"]}”, Top-5 neighbor Jaccard was {experiment["jaccard"]:.3f}; losses were {experiment["rows"][0]["final_loss"]:.3f} and {experiment["rows"][1]["final_loss"]:.3f}; vocabularies were {experiment["rows"][0]["vocabulary"]} and {experiment["rows"][1]["vocabulary"]}; the first-two-PC variance was {experiment["rows"][0]["pca_variance"]:.3f} and {experiment["rows"][1]["pca_variance"]:.3f}.'
        if experiment else "Run a Parameter Experiment to generate evidence for this question."
    )
    visualization = visualization or {"pca":{"points":[],"variance":0},"tsne":{"points":[],"perplexity":"n/a"}}
    return [
        {"question": "RQ1 · How do CBOW and Skip-Gram differ?", "answer": (
            f'CBOW used context→center training and took {cbow["seconds"]} s (final/average loss {cbow["final_loss"]:.3f}/{cbow["average_loss"]:.3f}); '
            f'Skip-Gram used center→context training and took {skipgram["seconds"]} s ({skipgram["final_loss"]:.3f}/{skipgram["average_loss"]:.3f}). '
            f'The mean Top-5 neighbor Jaccard over {len(neighbor_rows)} targets was {average_overlap:.3f}. Rare-word observations: {rare_summary or "no eligible shared rare words"}. '
            "These corpus results, rather than theory alone, determine which representation looks more useful here."
        )},
        {"question": "RQ2 · Do the embeddings show semantic meaning?", "answer": (
            f'Observed contextual-pair cosine scores were {pair_summary or "unavailable"}. Nearest-neighbor lists provide additional qualitative evidence. '
            "Word2Vec captures distributional/contextual similarity, not guaranteed dictionary synonymy, and cosine similarity is not accuracy."
        )},
        {"question": "RQ3 · How do PCA and t-SNE differ?", "answer": (
            f'This run plotted {len(visualization["pca"].get("points",[]))} words. PCA PC1+PC2 explained {visualization["pca"].get("variance",0):.3f} of normalized-vector variance; t-SNE used perplexity {visualization["tsne"].get("perplexity","n/a")}. '
            "PCA is a reproducible linear projection that exposes major global variance and meaningful PC axes; t-SNE is non-linear and emphasizes local neighborhoods. "
            "Both plots use the same normalized model vectors. Two-dimensional distance is only an approximation, and t-SNE clusters do not prove a correct semantic representation."
        )},
        {"question": "RQ4 · How do model parameters affect results?", "answer": rq4},
        {"question": "RQ5 · What limits a small corpus?", "answer": (
            f'This run used {cbow["documents"]} abstracts and {cbow["input_tokens"]} tokens. Around 1,000 abstracts can reveal domain-local co-occurrence patterns, '
            "but it is insufficient for complete language semantics. Rare terms receive little evidence; scientific terminology, domain/source bias, topic selection, and unstable neighbor rankings limit generalization."
        )},
    ]
