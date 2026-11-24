# =============================================================================
# python manage.py fetch_pubmed_glp1 [--count 20000]
# Project 5 的第二個資料來源：PubMed 上 GLP-1 相關的文獻（標題 + 摘要），作業建議的「疾病名稱，例如 GLP-1」
#
#   查詢：GLP-1 AND hasabstract[text] AND english[lang]（有摘要的英文文獻）
#   取最新的 N 篇：PubMed 的 esearch 一次最多只能拿 10,000 筆 PMID，所以從今年往回一年一年查，
#                  某一年超過 10,000 篇就再切成月份
#   下載：efetch 每次 200 篇，解析標題、摘要（有段落標籤的保留 BACKGROUND: 之類的標籤）、期刊、年份
#   docID：依 PMID 由小到大重新編號成 1–N（BSBI / SPIMI 依 docID 順序讀，和 Reuters 的 NEWID 一樣是連續的小整數）
#
# 輸出 data/indexing/pubmed/glp1.jsonl.gz（不進 git），每行一篇：
#   {"id": 1, "pmid": "12345678", "title": ..., "abstract": [段落...], "journal": ..., "year": "2024"}
# 連線沿用 search/importing/pmc_client.py 的 _get（憑證、重試、每秒最多約 3 次）；
# 請求只帶一般的 tool 名稱，不送 email 或 API key。
# =============================================================================
import gzip
import json
import urllib.parse
from datetime import date
from xml.etree import ElementTree as ET

from django.core.management.base import BaseCommand

from indexing.sources import PUBMED_FILE
from search.importing.pmc_client import EUTILS, _get

QUERY = 'GLP-1 AND hasabstract[text] AND english[lang]'
TOOL = 'ir-indexing-homework'
BATCH = 200


def _esearch(term, retmax=10000):
    params = urllib.parse.urlencode({'db': 'pubmed', 'term': term, 'retmode': 'json', 'retmax': retmax,
                                     'sort': 'pub_date', 'tool': TOOL})
    result = json.loads(_get(f'{EUTILS}/esearch.fcgi?{params}', timeout=60))['esearchresult']
    return int(result['count']), result.get('idlist', [])


def collect_pmids(count, log):
    pmids = {}                                  # dict 保留順序、去掉重複
    for year in range(date.today().year, 1979, -1):
        total, ids = _esearch(f'({QUERY}) AND {year}[pdat]')
        if total > 10000:                       # 一年超過 10,000 篇：切成月份
            ids = [p for month in range(12, 0, -1) for p in _esearch(f'({QUERY}) AND {year}/{month:02d}[pdat]')[1]]
        pmids.update(dict.fromkeys(ids))
        log(f'  {year}: {total:,} 篇，累計 {len(pmids):,}')
        if len(pmids) >= count * 1.03:          # 多拿 3%：少數文獻 efetch 後沒有摘要
            break
    return list(pmids)


def _text(node):
    return ' '.join(''.join(node.itertext()).split()) if node is not None else ''


def parse(xml):
    records = []
    for article in ET.fromstring(xml).iter('PubmedArticle'):
        pmid = article.findtext('./MedlineCitation/PMID', '').strip()
        meta = article.find('./MedlineCitation/Article')
        if not pmid or meta is None:
            continue
        paragraphs = []
        for part in meta.findall('./Abstract/AbstractText'):
            text = _text(part)
            if text:
                label = part.get('Label')
                paragraphs.append(f'{label}: {text}' if label else text)
        if not paragraphs:
            continue
        pub = meta.find('./Journal/JournalIssue/PubDate')
        year = (pub.findtext('Year') or (pub.findtext('MedlineDate') or '')[:4]) if pub is not None else ''
        records.append({'pmid': pmid, 'title': _text(meta.find('ArticleTitle')), 'abstract': paragraphs,
                        'journal': meta.findtext('./Journal/ISOAbbreviation') or meta.findtext('./Journal/Title') or '',
                        'year': year})
    return records


class Command(BaseCommand):
    help = 'Download recent PubMed GLP-1 abstracts for the Project 5 indexing experiments.'

    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=20000)

    def handle(self, *args, **options):
        count = options['count']
        log = lambda message: self.stdout.write(message)
        log(f'PubMed esearch: {QUERY}')
        pmids = collect_pmids(count, log)
        records = {}
        for start in range(0, len(pmids), BATCH):
            batch = pmids[start:start + BATCH]
            params = urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(batch), 'retmode': 'xml', 'tool': TOOL})
            for record in parse(_get(f'{EUTILS}/efetch.fcgi?{params}', timeout=120)):
                records[record['pmid']] = record
            if start // BATCH % 10 == 0:
                log(f'  efetch {start + len(batch):,}/{len(pmids):,}，有摘要的 {len(records):,} 篇')
            if len(records) >= count:
                break
        # 取最新的 count 篇後，依 PMID 由小到大編 docID 1–N
        chosen = sorted(records.values(), key=lambda r: (-int(r['year'] or 0), -int(r['pmid'])))[:count]
        chosen.sort(key=lambda r: int(r['pmid']))
        PUBMED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(PUBMED_FILE, 'wt', encoding='utf-8') as stream:
            for doc_id, record in enumerate(chosen, 1):
                stream.write(json.dumps({'id': doc_id, **record}, ensure_ascii=False) + '\n')
        years = sorted({r['year'] for r in chosen if r['year']})
        self.stdout.write(self.style.SUCCESS(f'{len(chosen):,} abstracts ({years[0]}–{years[-1]}) → {PUBMED_FILE}'))
