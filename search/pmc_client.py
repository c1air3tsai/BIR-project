import re
import json
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from functools import lru_cache
from xml.etree import ElementTree as ET
from django.conf import settings
import truststore
from http.client import IncompleteRead, RemoteDisconnected


class NCBITransientError(RuntimeError):
    """A temporary source failure that the import can retry without losing work."""

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "biomedir_student_ir"
_request_lock = threading.Lock()
_last_request = 0.0


@lru_cache(maxsize=1)
def _ssl_context():
    """Use native trust plus explicitly supplied, trusted local CA certificates."""
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.set_alpn_protocols(['http/1.1'])
    directory = Path(getattr(settings, 'NCBI_CA_DIR', Path(settings.BASE_DIR) / 'certs'))
    certificates = sorted({*directory.glob('*.crt'), *directory.glob('*.pem')})
    bundle = getattr(settings, 'NCBI_CA_BUNDLE', '')
    if bundle:
        certificates.append(Path(bundle))
    for certificate in certificates:
        try:
            context.load_verify_locations(cafile=str(certificate))
        except (OSError, ssl.SSLError) as exc:
            raise RuntimeError(f'Cannot load trusted CA file {certificate.name}. Use a PEM CA certificate.') from exc
    return context


def _certificate_error(exc):
    reason = getattr(exc, 'reason', exc)
    if isinstance(reason, ssl.SSLCertVerificationError) or 'CERTIFICATE_VERIFY_FAILED' in str(reason):
        raise RuntimeError(
            'SSL certificate verification failed. On Windows, run prepare_certs.ps1, '
            'then rebuild/restart Docker to load your trusted CA certificates. '
            'If it still fails, obtain the network proxy CA from your IT administrator.'
        ) from exc


def _get(url, timeout=30, min_interval=0.36):
    global _last_request
    source = "arXiv" if "export.arxiv.org/" in url else "NCBI"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BioMedIR-Student-Project/1.0"},
    )
    context = _ssl_context()
    for attempt in range(4):
        with _request_lock:
            delay = min_interval - (time.monotonic() - _last_request)
            if delay > 0:
                time.sleep(delay)
            _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                error = NCBITransientError if exc.code in {429, 500, 502, 503, 504} else RuntimeError
                raise error(f"{source} request failed (HTTP {exc.code}). Please try again.") from exc
        except ssl.SSLCertVerificationError as exc:
            _certificate_error(exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead, RemoteDisconnected) as exc:
            _certificate_error(exc)
            if attempt == 3:
                reason = getattr(exc, 'reason', exc)
                raise NCBITransientError(f"Cannot reach {source}: {reason}. Check your internet connection and try again.") from exc
        time.sleep(2 ** attempt)


def _params(values, email=None):
    values = dict(values, tool=TOOL)
    supplied_email = values.pop("email", None)
    contact = email or getattr(settings, "NCBI_EMAIL", "") or supplied_email
    if contact and contact != "student@example.com":
        values["email"] = contact
    if getattr(settings, "NCBI_API_KEY", ""):
        values["api_key"] = settings.NCBI_API_KEY
    return urllib.parse.urlencode(values)


def search_pubmed(query, retstart=0, retmax=200):
    """Search reproducibly for English records with an abstract, in relevance order."""
    params = _params({
        "db": "pubmed", "term": f"({query}) AND hasabstract[text] AND english[lang]",
        "retmode": "json", "retstart": retstart, "retmax": min(retmax, 1000),
        "sort": "relevance",
    })
    try:
        result = json.loads(_get(f"{EUTILS}/esearch.fcgi?{params}"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise NCBITransientError('NCBI returned incomplete search JSON. Retrying the search may help.') from exc
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    search = result.get("esearchresult", {})
    if search.get("errorlist"):
        raise RuntimeError("PubMed could not interpret this query. Try a simpler topic.")
    return list(dict.fromkeys(search.get("idlist", []))), int(search.get("count", 0))


def fetch_pubmed_batch(pmids):
    """Download at most 50 abstracts per request and split them into HW1 XML records."""
    if len(pmids) > 50:
        raise ValueError("Use batches of at most 50 PMIDs.")
    params = _params({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    data = _get(f"{EUTILS}/efetch.fcgi?{params}")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise NCBITransientError('NCBI returned incomplete batch XML. Retrying the download may help.') from exc
    if str(root.tag).split('}')[-1] != 'PubmedArticleSet':
        _validate_pubmed_xml(data)
    error = root.find('.//ERROR')
    if error is not None:
        raise RuntimeError('NCBI could not fetch these records: ' + ''.join(error.itertext()).strip())
    # A legitimate set may contain only book records, deleted IDs or no usable
    # English journal articles. Return zero records so the worker skips them.
    records = []
    for article in root.iter("PubmedArticle"):
        pmid = article.findtext("./MedlineCitation/PMID", "").strip()
        languages = [node.text for node in article.findall("./MedlineCitation/Article/Language")]
        if not pmid or (languages and "eng" not in languages):
            continue
        wrapper = ET.Element("PubmedArticleSet")
        wrapper.append(article)
        records.append((pmid, ET.tostring(wrapper, encoding="utf-8", xml_declaration=True)))
    return records


def normalize_pmcid(pmcid: str):
    """Normalize an explicit PMC identifier."""
    value = (pmcid or "").strip().upper()
    if not value.startswith("PMC"):
        value = "PMC" + value
    if not re.fullmatch(r"PMC\d+", value):
        raise ValueError("Invalid PMCID. Example: PMC8270360")
    return value


def normalize_identifier(identifier: str):
    """
    Return (kind, normalized_id) for a PMID or PMCID.

    Rules:
    - PMC1234567 -> ("pmc", "PMC1234567")
    - PMID:42724776 / PMID42724776 -> ("pubmed", "42724776")
    - 42724776 -> ("pubmed", "42724776")

    Bare digits are treated as PMID. A PMCID should include the PMC prefix.
    """
    value = (identifier or "").strip().upper()
    value = re.sub(r"\s+", "", value)

    if re.fullmatch(r"PMC\d+", value):
        return "pmc", value

    pmid_match = re.fullmatch(r"PMID:?([0-9]+)", value)
    if pmid_match:
        return "pubmed", pmid_match.group(1)

    if re.fullmatch(r"\d+", value):
        return "pubmed", value

    raise ValueError(
        "Invalid identifier. Enter a PMID such as 42724776 or a PMCID such as PMC12503546."
    )


def _validate_pmc_xml(data: bytes):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise NCBITransientError("NCBI returned incomplete or invalid XML; retrying the download may help.") from exc
    if root.find(".//article") is None and str(root.tag).split("}")[-1] != "article":
        raise RuntimeError("PMC article XML was not found for this PMCID")


def _validate_pubmed_xml(data: bytes):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise NCBITransientError("NCBI returned incomplete or invalid XML; retrying the download may help.") from exc
    has_article = any(str(node.tag).split("}")[-1] == "PubmedArticle" for node in root.iter())
    if not has_article:
        raise RuntimeError("PubMed article XML was not found for this PMID")


def download_pmc_xml(pmcid: str, output_dir: str, email: str = "student@example.com"):
    pmcid = normalize_pmcid(pmcid)
    params = _params({
        "db": "pmc",
        "id": pmcid,
        "rettype": "xml",
        "retmode": "xml",
        "tool": TOOL,
        "email": email,
    })
    data = _get(f"{EUTILS}/efetch.fcgi?{params}")
    _validate_pmc_xml(data)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pmcid}.xml"
    out_path.write_bytes(data)
    return str(out_path)


def download_pubmed_xml(pmid: str, output_dir: str, email: str = "student@example.com"):
    """Download PubMed XML for a PMID. This is sufficient for Title/Abstract metadata."""
    kind, normalized = normalize_identifier(pmid)
    if kind != "pubmed":
        raise ValueError("A PMID is required")

    params = _params({
        "db": "pubmed",
        "id": normalized,
        "retmode": "xml",
        "tool": TOOL,
        "email": email,
    })
    data = _get(f"{EUTILS}/efetch.fcgi?{params}")
    _validate_pubmed_xml(data)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"PMID{normalized}.xml"
    out_path.write_bytes(data)
    return str(out_path)


def download_article_xml(identifier: str, output_dir: str, email: str = "student@example.com"):
    """Fetch XML using either a PMCID or PMID."""
    kind, normalized = normalize_identifier(identifier)
    if kind == "pmc":
        return download_pmc_xml(normalized, output_dir, email=email)
    return download_pubmed_xml(normalized, output_dir, email=email)


def search_pmc(query: str, retmax: int = 5, email: str = "student@example.com"):
    params = _params({
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": max(1, min(int(retmax), 20)),
        "tool": TOOL,
        "email": email,
    })
    import json
    data = json.loads(_get(f"{EUTILS}/esearch.fcgi?{params}").decode("utf-8"))
    return [f"PMC{x}" for x in data.get("esearchresult", {}).get("idlist", [])]
