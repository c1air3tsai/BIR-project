# =============================================================================
# weighting/preprocessing.py
# Project 3 任務一：文字前處理（可開關 stopword removal 與 Porter stemming）
#
# 流程（文件與查詢都走同一條流程，確保在同一個詞彙空間）：
#   1. Tokenization：沿用 HW1 的生醫 tokenizer（search/text_processing.py），
#      GLP-1、COVID-19、0.05 這類詞會保持完整，獨立的標點不會變成 token（= 去標點）
#   2. Lowercase：tokenizer 內已做 casefold（比 lower() 更完整的小寫化）
#   3. 連字號統一：‐ ‑ – 三種連字號都換成一般的 "-"，讓 GLP‑1 與 GLP-1 視為同一詞
#   4. （可選）Stopword removal：使用 HW1 的 72 個 stopword
#   5. （可選）Stemming：NLTK Porter stemmer
#
# 被誰使用：weighting/index.py（建索引、處理查詢）、weighting/views.py（摘要高亮）
# =============================================================================
from dataclasses import dataclass

from search.text_processing import STOPWORDS, _stem_token, iter_token_matches

# 三種長得像連字號的字元統一成 "-"
_CONNECTORS = str.maketrans({'‐': '-', '‑': '-', '–': '-'})


# 前處理設定：兩個開關。frozen=True 讓它可以當作 dict key / 快取 key 使用
@dataclass(frozen=True)
class PreprocessConfig:
    remove_stopwords: bool = True
    stemming: bool = False

    # 給快取檔名與網址參數用的短代號，例如 "stop1_stem0"
    @property
    def key(self):
        return f'stop{int(self.remove_stopwords)}_stem{int(self.stemming)}'

    # 給網頁與 CSV 顯示用的說明文字
    @property
    def label(self):
        parts = ['tokenize + lowercase + remove punctuation']
        parts.append('remove stopwords' if self.remove_stopwords else 'keep stopwords')
        parts.append('Porter stemming' if self.stemming else 'no stemming')
        return ' → '.join(parts)


# 本專案對照實驗固定使用的前處理（控制變因）：移除 stopword、不做 stemming
DEFAULT_CONFIG = PreprocessConfig(remove_stopwords=True, stemming=False)


# 把單一個原始 token 轉成索引用的詞項；若被 stopword 過濾掉則回傳 None
def normalize_token(raw_token, config):
    token = raw_token.casefold().translate(_CONNECTORS)
    if config.remove_stopwords and token in STOPWORDS:
        return None
    if config.stemming:
        token = _stem_token(token)
    return token


# 把一段文字轉成詞項清單（文件與查詢共用）
def preprocess(text, config=DEFAULT_CONFIG):
    return [term for _, _, term in term_spans(text, config) if term]


# 回傳每個 token 在原文中的位置與處理後詞項：[(start, end, term 或 None), ...]
# 用途：搜尋結果的摘要片段要把「命中查詢的詞」標成高亮，需要知道原文位置
def term_spans(text, config=DEFAULT_CONFIG):
    text = text or ''
    # 取原文切片（與 HW1 tokenize() 相同做法），不用 match.group(0)
    return [(m.start(), m.end(), normalize_token(text[m.start():m.end()], config))
            for m in iter_token_matches(text)]
