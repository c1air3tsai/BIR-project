# =============================================================================
# indexing/queries.py
# Project 5 檢索測試的查詢（設定檔），依資料來源（sources.py）分開
#   每組兩個 terms，各做 single-term（兩個 term 各一次）、AND、OR，分別查 BSBI 與 SPIMI 的索引
#
#   QUERIES[來源]：主要的 5 組，符合該來源內容的查詢
#       reuters：1987 年財經新聞（報告的檢索測試表格用這 5 組）
#       pubmed ：GLP-1 文獻（藥物、療效、副作用）
#   EXAMPLE_QUERIES：作業列出的 5 組範例查詢，兩個來源都查，當作對照
#                    （在 Reuters 財經新聞裡兩個詞很少同時出現，例如 information AND retrieval 只有 2 篇）
#   TRACE_TERMS[來源]：中間結果要追蹤的 term，看它在各個 block 的局部 postings 與合併後的完整 postings
#                     （挑 df 小、但分散在好幾個 blocks 的詞，整條 postings 才放得進頁面）
#   DEFAULT_TERMS[來源]：Part 1 即時查詢的預設兩個詞
# =============================================================================
QUERIES = {
    'reuters': [
        ('Query 1', 'oil', 'price'),
        ('Query 2', 'wheat', 'export'),
        ('Query 3', 'crude', 'petroleum'),
        ('Query 4', 'interest', 'rate'),
        ('Query 5', 'trade', 'deficit'),
    ],
    'pubmed': [
        ('Query 1', 'semaglutide', 'weight'),
        ('Query 2', 'liraglutide', 'obesity'),
        ('Query 3', 'insulin', 'secretion'),
        ('Query 4', 'cardiovascular', 'outcomes'),
        ('Query 5', 'gastrointestinal', 'adverse'),
    ],
}

EXAMPLE_QUERIES = [
    ('Example 1', 'information', 'retrieval'),
    ('Example 2', 'artificial', 'intelligence'),
    ('Example 3', 'financial', 'market'),
    ('Example 4', 'medical', 'research'),
    ('Example 5', 'computer', 'science'),
]

TRACE_TERMS = {
    'reuters': ('retrieval', 'intelligence'),
    'pubmed': ('retrieval', 'parkinson'),
}

DEFAULT_TERMS = {
    'reuters': ('oil', 'price'),
    'pubmed': ('semaglutide', 'weight'),
}
