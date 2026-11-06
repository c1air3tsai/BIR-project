# =============================================================================
# reuters/corpus.py
# Project 4：讀取 Reuters-21578（UCI Machine Learning Repository 版本）
#
# 資料來源：https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection
#   壓縮檔內有官方 README.txt 與 reuters21578.tar.gz（22 個 reut2-0xx.sgm 檔）
#
# ModApte 切分（依官方 README 第 VIII.B 節）：
#   訓練集 9,603 篇：LEWISSPLIT="TRAIN" 且 TOPICS="YES"
#   測試集 3,299 篇：LEWISSPLIT="TEST"  且 TOPICS="YES"
#   其餘 8,676 篇不使用（unused）
#   README 特別交代：不要從 3,299 篇測試集中刪掉任何文件
#
# 本模組只負責「下載 → 安全解壓 → 解析 SGML → 存成 JSON 快取」，
# 前處理在 reuters/text.py，檢索在 reuters/retrieval.py。
# =============================================================================
import html
import io
import json
import os
import re
import tarfile
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

from django.conf import settings

UCI_URL = 'https://archive.ics.uci.edu/static/public/137/reuters+21578+text+categorization+collection.zip'

# 作業建議的 5 個主題
TOPICS = {
    'earn': '企業獲利與財務結果',
    'acq': '企業收購與合併',
    'crude': '原油',
    'grain': '穀物',
    'trade': '國際貿易',
}

_REUTERS = re.compile(r'<REUTERS([^>]*)>(.*?)</REUTERS>', re.S)
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def data_dir():
    return Path(settings.BASE_DIR) / 'data' / 'reuters'


def raw_dir():
    return data_dir() / 'raw' / 'extracted'


def cache_path():
    return data_dir() / 'corpus.json'


# 從 UCI 下載壓縮檔並安全解壓（檢查每個檔名不會跳出目標資料夾）
def download_and_extract(force=False):
    target = raw_dir()
    if not force and any(target.glob('reut2-*.sgm')):
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = data_dir() / 'raw' / 'reuters21578.zip'
    if force or not archive.is_file():
        with urllib.request.urlopen(UCI_URL, timeout=120) as response:
            archive.write_bytes(response.read())
    root = os.path.abspath(target)
    with zipfile.ZipFile(archive) as outer:
        (target / 'README.txt').write_bytes(outer.read('README.txt'))
        with tarfile.open(fileobj=io.BytesIO(outer.read('reuters21578.tar.gz'))) as inner:
            for member in inner.getmembers():
                destination = os.path.abspath(os.path.join(root, member.name))
                if not destination.startswith(root + os.sep) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsafe archive member: {member.name}')
            inner.extractall(root)
    return target


# 取出某個標籤內的文字（沒有就回傳空字串），並把 &lt; 這類 HTML 編碼還原
def _tag(body, name):
    match = re.search(rf'<{name}>(.*?)</{name}>', body, re.S)
    return html.unescape(match.group(1)).strip() if match else ''


# 新聞內文最後固定有 "Reuter" 署名與控制字元，去掉
def _clean_body(text):
    text = text.replace('\x03', '')
    return re.sub(r'\s*\bReuter\s*$', '', text.strip(), flags=re.IGNORECASE).strip()


# ModApte 切分：train / test / unused
def modapte_split(attrs):
    if attrs.get('TOPICS') != 'YES':
        return 'unused'
    return {'TRAIN': 'train', 'TEST': 'test'}.get(attrs.get('LEWISSPLIT'), 'unused')


# 解析全部 22 個 SGML 檔，回傳文件清單
def parse_sgml(directory):
    documents = []
    for path in sorted(Path(directory).glob('reut2-*.sgm')):
        raw = path.read_bytes().decode('latin-1')
        for match in _REUTERS.finditer(raw):
            attrs = dict(_ATTR.findall(match.group(1)))
            body = match.group(2)
            topics_block = re.search(r'<TOPICS>(.*?)</TOPICS>', body, re.S)
            documents.append({
                'id': int(attrs['NEWID']),
                'split': modapte_split(attrs),
                'topics': re.findall(r'<D>(.*?)</D>', topics_block.group(1)) if topics_block else [],
                'date': _tag(body, 'DATE')[:11],
                'title': _tag(body, 'TITLE'),
                'body': _clean_body(_tag(body, 'BODY')),
            })
    return documents


# 解析一次後存成 JSON；之後直接讀快取
def build_cache(force=False):
    path = cache_path()
    if path.is_file() and not force:
        return path
    documents = parse_sgml(download_and_extract())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(documents, ensure_ascii=False), encoding='utf-8')
    load_documents.cache_clear()
    return path


@lru_cache(maxsize=1)
def load_documents():
    return json.loads(build_cache().read_text(encoding='utf-8'))


def split_documents(split):
    return [doc for doc in load_documents() if doc['split'] == split]


# 文件的檢索文字：標題 + 內文（約 9% 的新聞沒有內文，只有標題）
def document_text(doc):
    return f"{doc['title']}\n{doc['body']}".strip()
