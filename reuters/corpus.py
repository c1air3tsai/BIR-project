# =============================================================================
# reuters/corpus.py
# Project 4：讀取 Reuters-21578（UCI Machine Learning Repository 版本）
#
# 資料來源：https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection
#   壓縮檔內有官方 README.txt 與 reuters21578.tar.gz（22 個 reut2-0xx.sgm 檔）
#
# 官方 README 第 VIII 節定義了三種切分（SPLITS），網頁可以切換：
#   ModApte（VIII.B，預設）：訓練集 9,603 篇 = LEWISSPLIT="TRAIN" 且 TOPICS="YES"
#                            測試集 3,299 篇 = LEWISSPLIT="TEST"  且 TOPICS="YES"，其餘 8,676 篇不使用
#                            README 推薦用這個切分做 TOPICS 研究，作業也指定它
#   ModLewis（VIII.A）：訓練集 13,625 篇 = TRAIN、測試集 6,188 篇 = TEST（TOPICS 為 YES 或 NO 都算），
#                       NOT-USED 或 TOPICS="BYPASS" 的 1,765 篇不使用
#   ModHayes（VIII.C）：訓練集 20,856 篇 = CGISPLIT="TRAINING-SET"、測試集 722 篇 = CGISPLIT="PUBLISHED-TESTSET"
#   README 特別交代：不要從測試集中刪掉任何文件
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
DATASET_NAME = 'Reuters-21578'
# JSON 快取的格式版本：欄位改變時加 1，舊快取會自動重新解析
CACHE_VERSION = 2

# 作業建議的 5 個主題
TOPICS = {
    'earn': '企業獲利與財務結果',
    'acq': '企業收購與合併',
    'crude': '原油',
    'grain': '穀物',
    'trade': '國際貿易',
}

# 其他常見 TOPICS 標籤的中文名稱（新聞列表與內頁顯示用；沒列到的直接顯示原本的代號）
TOPIC_NAMES = {
    **TOPICS,
    'money-fx': '外匯', 'interest': '利率', 'wheat': '小麥', 'ship': '航運', 'corn': '玉米', 'dlr': '美元',
    'oilseed': '油籽', 'money-supply': '貨幣供給', 'sugar': '糖', 'gnp': '國民生產毛額', 'coffee': '咖啡',
    'veg-oil': '植物油', 'gold': '黃金', 'nat-gas': '天然氣', 'soybean': '黃豆', 'bop': '國際收支',
    'livestock': '家畜', 'cpi': '消費者物價', 'reserves': '外匯存底', 'copper': '銅', 'cocoa': '可可',
    'jobs': '就業', 'carcass': '肉品', 'yen': '日圓', 'rice': '稻米', 'cotton': '棉花', 'iron-steel': '鋼鐵',
    'ipi': '工業生產', 'alum': '鋁', 'gas': '汽油', 'barley': '大麥', 'meal-feed': '飼料', 'rubber': '橡膠',
    'palm-oil': '棕櫚油', 'silver': '白銀', 'tin': '錫', 'zinc': '鋅', 'wpi': '躉售物價', 'retail': '零售',
    'housing': '房市', 'pet-chem': '石化', 'sorghum': '高粱', 'strategic-metal': '戰略金屬', 'lead': '鉛',
    'orange': '柳橙', 'heat': '燃料油', 'fuel': '燃料', 'income': '個人所得', 'hog': '豬', 'lei': '領先指標',
    'soy-oil': '黃豆油', 'soy-meal': '黃豆粉', 'rapeseed': '油菜籽', 'oat': '燕麥', 'tea': '茶', 'lumber': '木材',
    'nickel': '鎳', 'platinum': '鉑', 'l-cattle': '活牛', 'potato': '馬鈴薯', 'wool': '羊毛', 'jet': '航空燃油',
    'propane': '丙烷', 'naphtha': '石腦油', 'cpu': '產能利用率', 'instal-debt': '分期債務', 'dmk': '德國馬克',
    'stg': '英鎊', 'sunseed': '葵花籽', 'groundnut': '花生', 'coconut': '椰子', 'inventories': '存貨',
}

# 官方 README 第 VIII 節的三種切分（網頁的切分選單與說明文字都從這裡來）
SPLITS = {
    'modapte': {'label': 'ModApte', 'section': 'VIII.B',
                'train': 'LEWISSPLIT = TRAIN 且 TOPICS = YES', 'test': 'LEWISSPLIT = TEST 且 TOPICS = YES',
                'unused': '其餘（TOPICS = NO / BYPASS，或 LEWISSPLIT = NOT-USED）',
                'summary': '只用有 TOPICS 標註的新聞。README 推薦用於主題研究，作業也指定這個（預設）。'},
    'modlewis': {'label': 'ModLewis', 'section': 'VIII.A',
                 'train': 'LEWISSPLIT = TRAIN（TOPICS = YES 或 NO）', 'test': 'LEWISSPLIT = TEST（TOPICS = YES 或 NO）',
                 'unused': 'LEWISSPLIT = NOT-USED 或 TOPICS = BYPASS',
                 'summary': '連 TOPICS = NO（沒有主題標註）的新聞也放進來，測試集多出約 2,900 篇沒有標籤的新聞。'},
    'modhayes': {'label': 'ModHayes', 'section': 'VIII.C',
                 'train': 'CGISPLIT = TRAINING-SET', 'test': 'CGISPLIT = PUBLISHED-TESTSET', 'unused': '無',
                 'summary': '重現 1990 年 CONSTRUE 系統的測試集，只有 722 篇；README 指出主題分布不尋常、有近似重複。'},
}
DEFAULT_SPLIT = 'modapte'
SPLIT_PARTS = {'train': '訓練集', 'test': '測試集', 'unused': '未使用'}

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


# 三種切分的規則：輸入 <REUTERS> 標籤的屬性，回傳 'train' / 'test' / 'unused'
def modapte_split(attrs):
    if attrs.get('TOPICS') != 'YES':
        return 'unused'
    return {'TRAIN': 'train', 'TEST': 'test'}.get(attrs.get('LEWISSPLIT'), 'unused')


def modlewis_split(attrs):
    if attrs.get('TOPICS') == 'BYPASS':
        return 'unused'
    return {'TRAIN': 'train', 'TEST': 'test'}.get(attrs.get('LEWISSPLIT'), 'unused')


def modhayes_split(attrs):
    return {'TRAINING-SET': 'train', 'PUBLISHED-TESTSET': 'test'}.get(attrs.get('CGISPLIT'), 'unused')


SPLIT_RULES = {'modapte': modapte_split, 'modlewis': modlewis_split, 'modhayes': modhayes_split}


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
                'splits': {name: rule(attrs) for name, rule in SPLIT_RULES.items()},
                'attrs': {key: attrs.get(key, '') for key in ('TOPICS', 'LEWISSPLIT', 'CGISPLIT')},
                'topics': re.findall(r'<D>(.*?)</D>', topics_block.group(1)) if topics_block else [],
                'date': _tag(body, 'DATE')[:11],
                'title': _tag(body, 'TITLE'),
                'body': _clean_body(_tag(body, 'BODY')),
            })
    return documents


def _cache_is_current(path):
    try:
        with open(path, encoding='utf-8') as stream:
            return stream.read(32).startswith(f'{{"version": {CACHE_VERSION},')
    except OSError:
        return False


# 解析一次後存成 JSON（{"version": ..., "documents": [...]}）；之後直接讀快取，格式過舊會重新解析
def build_cache(force=False):
    path = cache_path()
    if not force and _cache_is_current(path):
        return path
    documents = parse_sgml(download_and_extract())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'version': CACHE_VERSION, 'documents': documents}, ensure_ascii=False),
                    encoding='utf-8')
    load_documents.cache_clear()
    return path


@lru_cache(maxsize=1)
def load_documents():
    return json.loads(build_cache().read_text(encoding='utf-8'))['documents']


@lru_cache(maxsize=1)
def documents_by_id():
    return {doc['id']: doc for doc in load_documents()}


# 某個切分的某一部分：part = 'train' / 'test' / 'unused'
def split_documents(part, split=DEFAULT_SPLIT):
    return [doc for doc in load_documents() if doc['splits'][split] == part]


# 每個切分的 train / test / unused 篇數：{'modapte': {'train': 9603, 'test': 3299, 'unused': 8676}, ...}
@lru_cache(maxsize=1)
def split_counts():
    counts = {name: dict.fromkeys(SPLIT_PARTS, 0) for name in SPLITS}
    for doc in load_documents():
        for name, part in doc['splits'].items():
            counts[name][part] += 1
    return counts


# 文件的檢索文字：標題 + 內文（約 9% 的新聞沒有內文，只有標題）
def document_text(doc):
    return f"{doc['title']}\n{doc['body']}".strip()
