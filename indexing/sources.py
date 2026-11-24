# =============================================================================
# indexing/sources.py
# Project 5 的兩個資料來源（網頁上方可以切換，一次顯示一個來源的結果）
#
#   reuters  Reuters-21578 全部 21,578 篇新聞（標題 + 內文），docID = NEWID；沿用 Project 4 的資料
#   pubmed   PubMed 上 GLP-1 相關的 20,000 篇文獻（標題 + 摘要），docID = 依 PMID 排序後的 1–20,000
#            （python manage.py fetch_pubmed_glp1 下載到 data/indexing/pubmed/glp1.jsonl.gz）
#
# 兩個來源用同一套前處理、同一套 BSBI / SPIMI / Baseline 程式，只差在文件、查詢（queries.py）與結果放在哪裡：
#   Reuters：最終索引 data/indexing/full/、實驗結果 docs/project5_results/（第一次做的位置，不搬動）
#   PubMed ：最終索引 data/indexing/pubmed/full/、實驗結果 docs/project5_results/pubmed/
# =============================================================================
import gzip
import json
from functools import lru_cache
from pathlib import Path

from django.conf import settings

BASE = Path(settings.BASE_DIR)
PUBMED_FILE = BASE / 'data' / 'indexing' / 'pubmed' / 'glp1.jsonl.gz'
DEFAULT_SOURCE = 'reuters'

SOURCES = {
    'reuters': {'key': 'reuters', 'label': 'Reuters-21578', 'kicker': 'REUTERS-21578', 'unit': '篇新聞',
                'results_dir': BASE / 'docs' / 'project5_results', 'work_dir': BASE / 'data' / 'indexing'},
    'pubmed': {'key': 'pubmed', 'label': 'PubMed GLP-1', 'kicker': 'PUBMED · GLP-1', 'unit': '篇摘要',
               'results_dir': BASE / 'docs' / 'project5_results' / 'pubmed',
               'work_dir': BASE / 'data' / 'indexing' / 'pubmed'},
}


def get(source):
    return SOURCES.get(source) or SOURCES[DEFAULT_SOURCE]


def available(source):
    return source == 'reuters' or (source == 'pubmed' and PUBMED_FILE.is_file())


# ---------------------------------------------------------------------------
# PubMed GLP-1 的文件：{docID: {'id', 'pmid', 'title', 'abstract': [段落], 'journal', 'year'}}
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def pubmed_documents():
    if not PUBMED_FILE.is_file():
        raise FileNotFoundError(f'{PUBMED_FILE} 不存在，先執行 python manage.py fetch_pubmed_glp1')
    with gzip.open(PUBMED_FILE, 'rt', encoding='utf-8') as stream:
        return {record['id']: record for record in map(json.loads, stream)}


# 建索引用的文字：標題 + 摘要（和 Reuters 的「標題 + 內文」對應）
def pubmed_text(record):
    return '\n'.join([record['title'], *record['abstract']]).strip()
