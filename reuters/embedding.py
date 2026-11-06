# =============================================================================
# reuters/embedding.py
# Project 4：用 gensim 在 Reuters 上訓練 Word2Vec（CBOW 與 Skip-Gram）
#
#   CBOW（sg=0）：用「上下文的詞」預測「中間的詞」
#                 例：[shipment, of, ___, to, japan] → wheat
#   Skip-Gram（sg=1）：用「中間的詞」預測「上下文的每個詞」
#                 例：wheat → shipment, of, to, japan
#
# 訓練語料（控制變因：兩種架構用完全相同的句子與參數）：
#   每種切分（corpus.SPLITS：ModApte / ModLewis / ModHayes）各自用自己的訓練集訓練
#   'train'        該切分的訓練集（預設；ModApte 是 9,603 篇）
#   'train_unused' 訓練集 + 未使用的文件（README 說 unused 文件可用來收集統計資訊），
#                  用來觀察「語料變大」對詞向量與檢索的影響；ModHayes 沒有未使用的文件
#   測試集不拿來訓練，避免詞向量「看過」要被檢索的新聞
#
# 模型存在 data/reuters/models/<切分>_<語料>_<架構>.model，旁邊的 .json 記錄參數與每個 epoch 的 loss；
# 參數或語料改變時會自動重新訓練。
# =============================================================================
import hashlib
import json
import time
from functools import lru_cache

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from .corpus import data_dir, document_text, load_documents
from .text import sentences

ARCHITECTURES = {'cbow': 'CBOW', 'skipgram': 'Skip-Gram'}
CORPORA = {'train': '訓練集', 'train_unused': '訓練集 + 未使用'}

# 兩種架構共用的參數（小型語料常見設定；seed + 單一 worker 讓結果可重現）
PARAMS = {
    'vector_size': 100,   # 詞向量維度
    'window': 5,          # 上下文視窗：中心詞左右各 5 個詞
    'min_count': 3,       # 出現少於 3 次的詞不學向量
    'negative': 5,        # negative sampling：每個正例配 5 個隨機負例
    'sample': 1e-3,       # 高頻詞降採樣
    'epochs': 20,
    'seed': 42,
    'workers': 1,
}


def model_dir():
    return data_dir() / 'models'


# 某個切分、某個語料的訓練句子
def training_sentences(split, corpus='train'):
    parts = {'train'} if corpus == 'train' else {'train', 'unused'}
    result = []
    for doc in load_documents():
        if doc['splits'][split] in parts:
            result.extend(sentences(document_text(doc)))
    return result


# 參數 + 語料內容的指紋：任何一個改變都要重新訓練
def _fingerprint(split, corpus, architecture, sents):
    digest = hashlib.sha256(json.dumps([split, corpus, architecture, PARAMS], sort_keys=True).encode())
    digest.update(str(len(sents)).encode())
    digest.update(' '.join(' '.join(s) for s in sents[:200]).encode('utf-8'))
    return digest.hexdigest()[:16]


# gensim 的 get_latest_training_loss() 是累積值，每個 epoch 結束時相減得到該 epoch 的 loss
class _LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.previous = 0.0
        self.losses = []

    def on_epoch_end(self, model):
        total = model.get_latest_training_loss()
        self.losses.append(total - self.previous)
        self.previous = total


def _paths(split, corpus, architecture):
    base = model_dir() / f'{split}_{corpus}_{architecture}'
    return base.with_suffix('.model'), base.with_suffix('.json')


# 訓練（或讀取快取）一個模型；回傳 (gensim 模型, 訓練紀錄)
def train(split, corpus='train', architecture='cbow', force=False):
    sents = training_sentences(split, corpus)
    fingerprint = _fingerprint(split, corpus, architecture, sents)
    model_path, meta_path = _paths(split, corpus, architecture)
    if not force and model_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('fingerprint') == fingerprint:
            return Word2Vec.load(str(model_path)), meta
    logger = _LossLogger()
    started = time.perf_counter()
    model = Word2Vec(sentences=sents, sg=1 if architecture == 'skipgram' else 0,
                     compute_loss=True, callbacks=[logger], **PARAMS)
    meta = {'fingerprint': fingerprint, 'split': split, 'corpus': corpus, 'architecture': architecture, 'params': PARAMS,
            'sentences': len(sents), 'tokens': sum(len(s) for s in sents),
            'vocabulary': len(model.wv), 'seconds': round(time.perf_counter() - started, 1),
            'epoch_losses': logger.losses}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return model, meta


# 網頁與實驗用：讀取已訓練好的詞向量（KeyedVectors）與紀錄，記在記憶體
# （3 種切分 × 2 種架構，再加上 train + unused 的模型）
@lru_cache(maxsize=12)
def load(split, corpus='train', architecture='cbow'):
    model, meta = train(split, corpus, architecture)
    return model.wv, meta


# 一個詞的 Top-k 鄰近詞（cosine similarity，使用完整 100 維向量）
def neighbors(word, split, architecture='cbow', topn=5, corpus='train'):
    vectors, _ = load(split, corpus, architecture)
    word = word.strip().casefold()
    if word not in vectors.key_to_index:
        return None
    return [{'word': other, 'cosine': float(score)} for other, score in vectors.most_similar(word, topn=topn)]
