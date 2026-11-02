# =============================================================================
# weighting/schemes.py
# Project 3 任務二：各種 Term Weighting 的數學實作（純 numpy / scipy.sparse）
#
# 這個檔案只做「數學」，不碰資料庫，所以單元測試可以直接拿小矩陣來手算驗證。
#
# 記號：
#   N      = 文件總數
#   tf     = tf(t,d)，詞 t 在文件 d 出現的次數（count matrix 的一格）
#   df     = df(t)，包含詞 t 的文件數
#   idf    = log10(N / df)
#   counts = 文件 × 詞彙 的稀疏矩陣（scipy CSR），每格就是 tf
#
# 支援的方法（METHODS）：
#   raw_tf    w = tf
#   log_tf    w = 1 + log10(tf)，tf = 0 時為 0
#   idf       w = idf（只要詞有出現，就給 idf；沒出現為 0）
#   tfidf     w = tf × idf
#   logtfidf  w = (1 + log10(tf)) × idf
#   bm25      額外比較用的 modified TF-IDF（Okapi BM25 的詞權重形式）
#
# 每種方法都可以選擇「有 / 無 L2 normalization」：
#   有 L2：文件向量與查詢向量都除以自己的 L2 長度，分數 = 兩個單位向量的內積 = cosine similarity
#   無 L2：直接用原始權重向量做內積 q · d，長文件（權重總量大）容易拿到較高分
# =============================================================================
import numpy as np
from scipy import sparse

# 方法代號 → 顯示名稱（順序就是網頁、CSV、圖表的顯示順序）
METHODS = {
    'raw_tf': 'Raw TF',
    'log_tf': 'Log TF',
    'idf': 'IDF',
    'tfidf': 'TF-IDF',
    'logtfidf': 'Log TF-IDF',
    'bm25': 'BM25 (modified)',
}

# 作業要求的五種必要方法；bm25 是額外加的比較組
REQUIRED_METHODS = ('raw_tf', 'log_tf', 'idf', 'tfidf', 'logtfidf')

# 網頁與報告上顯示的公式（英文，網頁可切換中英文介面）
FORMULAS = {
    'raw_tf': 'w = tf(t,d)',
    'log_tf': 'w = 1 + log10(tf) if tf > 0, else 0',
    'idf': 'w = log10(N / df) if tf > 0, else 0',
    'tfidf': 'w = tf × log10(N / df)',
    'logtfidf': 'w = (1 + log10(tf)) × log10(N / df)',
    'bm25': 'w = idf_BM25 × tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))',
}

# BM25 的兩個參數（教科書常用值）
BM25_K1 = 1.2
BM25_B = 0.75


# df(t)：每一欄（每個詞）有幾個文件的 tf > 0
def document_frequency(counts):
    return np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int64)


# cf(t)：collection frequency，詞在整個語料出現的總次數 = 每一欄 tf 加總
def collection_frequency(counts):
    return np.asarray(counts.sum(axis=0)).ravel().astype(np.int64)


# idf(t) = log10(N / df(t))；df = 0（詞不存在）時定義為 0，避免除以 0
def inverse_document_frequency(df, n_documents):
    df = np.asarray(df, dtype=np.float64)
    idf = np.zeros_like(df)
    present = df > 0
    idf[present] = np.log10(n_documents / df[present])
    return idf


# BM25 版本的 idf：log10(1 + (N − df + 0.5) / (df + 0.5))，永遠 ≥ 0
def bm25_idf(df, n_documents):
    df = np.asarray(df, dtype=np.float64)
    return np.log10(1.0 + (n_documents - df + 0.5) / (df + 0.5))


# 對稀疏矩陣「有值的格子」套用函數 f，0 的格子保持 0（因為 f 只作用在 tf > 0 的地方）
def _map_nonzero(matrix, function):
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    result.data = function(result.data)
    result.eliminate_zeros()
    return result


# 每一欄乘上對應的係數（例如乘上 idf 向量）：W = M · diag(column_weights)
def _scale_columns(matrix, column_weights):
    return sparse.csr_matrix(matrix @ sparse.diags(column_weights))


# 計算整個「文件 × 詞彙」權重矩陣
#   counts       文件的 tf 矩陣（CSR）
#   method       METHODS 其中一個代號
#   idf          每個詞的 idf 向量（由整個語料算出）
#   doc_lengths  每篇文件長度 |d|（只有 bm25 用得到）
#   avg_length   平均文件長度 avgdl（只有 bm25 用得到）
#   bm25_idf_values  每個詞的 BM25 idf（只有 bm25 用得到）
def weight_matrix(counts, method, idf, doc_lengths=None, avg_length=None, bm25_idf_values=None):
    if method == 'raw_tf':
        # w = tf
        return sparse.csr_matrix(counts, dtype=np.float64)
    if method == 'log_tf':
        # w = 1 + log10(tf)，只對 tf > 0 的格子計算
        return _map_nonzero(counts, lambda tf: 1.0 + np.log10(tf))
    if method == 'idf':
        # w = idf：先把 tf 變成 0/1（有出現 = 1），再乘上 idf
        binary = _map_nonzero(counts, np.ones_like)
        return _scale_columns(binary, idf)
    if method == 'tfidf':
        # w = tf × idf
        return _scale_columns(sparse.csr_matrix(counts, dtype=np.float64), idf)
    if method == 'logtfidf':
        # w = (1 + log10 tf) × idf
        return _scale_columns(_map_nonzero(counts, lambda tf: 1.0 + np.log10(tf)), idf)
    if method == 'bm25':
        return _bm25_matrix(counts, doc_lengths, avg_length, bm25_idf_values)
    raise ValueError(f'Unknown weighting method: {method}')


# BM25 文件權重：w = idf_BM25 × tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
# tf 越大，權重越接近上限 idf·(k1+1)（飽和）；文件越長，分母越大（長度懲罰）
def _bm25_matrix(counts, doc_lengths, avg_length, idf_values):
    matrix = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    lengths = np.asarray(doc_lengths, dtype=np.float64)
    # 每個非零格屬於哪一列（哪一篇文件）
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    tf = matrix.data
    length_factor = 1.0 - BM25_B + BM25_B * lengths[rows] / (avg_length or 1.0)
    matrix.data = tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * length_factor)
    return _scale_columns(matrix, idf_values)


# 查詢向量：使用「和文件相同」的方法與同一組 idf
#   query_counts 為長度 V 的 numpy 陣列（查詢中每個詞出現幾次）
#   BM25 的查詢端一般只用查詢詞次數（raw tf），不做長度懲罰
def query_weights(query_counts, method, idf, bm25_idf_values=None):
    tf = np.asarray(query_counts, dtype=np.float64)
    present = tf > 0
    log_tf = np.zeros_like(tf)
    log_tf[present] = 1.0 + np.log10(tf[present])
    if method == 'raw_tf':
        return tf
    if method == 'log_tf':
        return log_tf
    if method == 'idf':
        return present * idf
    if method == 'tfidf':
        return tf * idf
    if method == 'logtfidf':
        return log_tf * idf
    if method == 'bm25':
        return tf
    raise ValueError(f'Unknown weighting method: {method}')


# 每一列（每篇文件）的 L2 長度：||d||2 = sqrt(Σ w²)
def row_l2_norms(matrix):
    return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())


# 把每一列除以自己的 L2 長度，變成單位向量；長度為 0 的列保持 0
def l2_normalize_rows(matrix):
    norms = row_l2_norms(matrix)
    inverse = np.zeros_like(norms)
    inverse[norms > 0] = 1.0 / norms[norms > 0]
    return sparse.csr_matrix(sparse.diags(inverse) @ matrix)


# 單一向量的 L2 正規化：v / ||v||2
def l2_normalize_vector(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


# 計算查詢與所有文件的分數
#   use_l2 = True ：score = (q / ||q||) · (d / ||d||) = cosine similarity
#   use_l2 = False：score = q · d（原始內積，不做長度正規化）
# doc_matrix 傳進來時，若 use_l2 為 True 應該已經是正規化過的矩陣（見 index.py 的快取）
def score_documents(doc_matrix, query_vector, use_l2):
    query = l2_normalize_vector(query_vector) if use_l2 else query_vector
    return np.asarray(doc_matrix @ query).ravel()


# 依分數由高到低排序，只保留分數 > 0 的文件；同分時用文件順序（較小 index 在前）
# 回傳 [(row_index, score), ...]
def rank_scores(scores, limit=10):
    candidates = np.flatnonzero(scores > 0)
    # np.lexsort 以最後一個 key 為主排序：先比 -score（高分在前），再比 row index
    order = candidates[np.lexsort((candidates, -scores[candidates]))]
    if limit is not None:
        order = order[:limit]
    return [(int(row), float(scores[row])) for row in order]


# 兩向量的 cosine similarity：cos(q, d) = (q · d) / (||q|| ||d||)
# （給單元測試與網頁說明使用；大量計算用上面的矩陣版本）
def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0
