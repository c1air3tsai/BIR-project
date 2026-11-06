# =============================================================================
# reuters/text.py
# Project 4：Reuters 新聞的前處理（TF-IDF 與 Word2Vec 共用同一套，詞彙才一致）
#
#   1. Tokenization + lowercase：沿用 HW1 的 tokenizer（search/text_processing.py）
#   2. 去標點：tokenizer 本身不產生標點 token
#   3. 移除 stopword：HW1 的 72 個 stopword
#   4. 移除「沒有英文字母」的 token（純數字、金額，如 12、1.5、87）：
#      財報新聞（earn）有大量數字，對詞向量的語意沒有幫助，只會增加雜訊
#   5. 不做 stemming：保留原詞形，鄰近詞表比較好解讀（profit / profits 分開）
#
# 另外提供切句：Word2Vec 以「句子」為訓練單位，上下文視窗不跨句。
# Reuters 內文會在句子中間換行，所以先把換行合併成空白，再用 HW1 的規則切句。
# =============================================================================
import re

from search.text_processing import STOPWORDS, split_sentences, tokenize

_HAS_LETTER = re.compile(r'[^\W\d_]')


# 一段文字 → 詞項清單（檢索與 Word2Vec 共用）
def preprocess(text):
    return [token for token in tokenize(text)
            if token not in STOPWORDS and _HAS_LETTER.search(token)]


# 一篇新聞 → 句子清單，每句是詞項清單（Word2Vec 訓練資料）
def sentences(text):
    flat = re.sub(r'\s+', ' ', text or '').strip()
    result = []
    for sentence in split_sentences(flat):
        terms = preprocess(sentence)
        if len(terms) >= 2:
            result.append(terms)
    return result
