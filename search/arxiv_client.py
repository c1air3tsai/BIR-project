"""Rate-limited Atom metadata import; no PDF/body download and no invented PMID."""
import re
import time
from urllib.parse import urlencode
from xml.etree import ElementTree as ET
from .pmc_client import _get, NCBITransientError

_last_request=0.0
NS={'atom':'http://www.w3.org/2005/Atom','open':'http://a9.com/-/spec/opensearch/1.1/','arxiv':'http://arxiv.org/schemas/atom'}

def search_arxiv(query, retstart=0, retmax=100):
    global _last_request
    delay=3.1-(time.monotonic()-_last_request)
    if delay>0: time.sleep(delay)
    _last_request=time.monotonic()
    # Always require a CS classification, including cross-listed records.
    expression=f'({query}) AND cat:cs.*' if re.search(r'\b(?:cat|all|ti|abs):',query) else f'all:{query} AND cat:cs.*'
    params=urlencode({'search_query':expression,'start':retstart,'max_results':min(retmax,100),'sortBy':'relevance','sortOrder':'descending'})
    try:root=ET.fromstring(_get('https://export.arxiv.org/api/query?'+params, timeout=60, min_interval=3.1))
    except ET.ParseError as exc:raise NCBITransientError('arXiv returned incomplete Atom XML; retrying may help.') from exc
    records=[]
    for entry in root.findall('atom:entry',NS):
        identifier=entry.findtext('atom:id','',NS)
        if '/api/errors' in identifier:raise RuntimeError(entry.findtext('atom:summary','arXiv rejected the query.',NS))
        raw=identifier.split('/abs/')[-1]
        arxiv_id=re.sub(r'v\d+$','',raw)
        if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})',arxiv_id):continue
        title=' '.join(entry.findtext('atom:title','',NS).split())
        abstract=' '.join(entry.findtext('atom:summary','',NS).split())
        if not abstract:continue
        # Adapter XML is explicitly marked by the arxiv ArticleId; PMID stays empty.
        wrapper=ET.Element('PubmedArticleSet');article=ET.SubElement(wrapper,'PubmedArticle');citation=ET.SubElement(article,'MedlineCitation');body=ET.SubElement(citation,'Article')
        ET.SubElement(body,'ArticleTitle').text=title
        abstract_node=ET.SubElement(body,'Abstract');ET.SubElement(abstract_node,'AbstractText').text=abstract
        ET.SubElement(body,'Language').text='eng'
        journal=ET.SubElement(body,'Journal');ET.SubElement(journal,'Title').text='arXiv (preprint)'
        issue=ET.SubElement(journal,'JournalIssue');date=ET.SubElement(issue,'PubDate');ET.SubElement(date,'Year').text=entry.findtext('atom:published','',NS)[:4]
        authors=ET.SubElement(body,'AuthorList')
        for name in entry.findall('atom:author/atom:name',NS):ET.SubElement(ET.SubElement(authors,'Author'),'CollectiveName').text=name.text
        ids=ET.SubElement(ET.SubElement(article,'PubmedData'),'ArticleIdList')
        ET.SubElement(ids,'ArticleId',IdType='arxiv').text=arxiv_id
        doi=entry.findtext('arxiv:doi','',NS)
        if doi:ET.SubElement(ids,'ArticleId',IdType='doi').text=doi
        records.append((arxiv_id,ET.tostring(wrapper,encoding='utf-8',xml_declaration=True)))
    total=root.findtext('open:totalResults',str(len(records)),NS)
    return records,int(total)
