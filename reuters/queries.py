# =============================================================================
# reuters/queries.py
# Project 4：評估查詢集（設定檔，要新增或修改查詢直接改這裡）
#
# 相關性判斷（作業指定）：以 Reuters 的 topics 欄位為準
#   新聞的 topics 含有查詢對應的主題 → 相關（1），否則不相關（0）
#
# 每個主題 4 個查詢，分兩種類型：
#   keyword    用該主題新聞最常見的關鍵字（TF-IDF 的主場）
#   paraphrase 刻意換一種說法，避開最典型的關鍵字（例如用 petroleum / energy 代替 oil），
#              用來測試 Word Embedding 能不能補上「字不一樣就比對不到」的弱點
# =============================================================================

EVALUATION_QUERIES = [
    # earn：企業獲利與財務結果
    {'id': 'E1', 'topic': 'earn', 'type': 'keyword', 'text': 'quarterly net profit and earnings per share'},
    {'id': 'E2', 'topic': 'earn', 'type': 'keyword', 'text': 'dividend declared to shareholders'},
    {'id': 'E3', 'topic': 'earn', 'type': 'paraphrase', 'text': 'company income results for the year'},
    {'id': 'E4', 'topic': 'earn', 'type': 'paraphrase', 'text': 'firm reports bigger losses than last year'},
    # acq：企業收購與合併
    {'id': 'A1', 'topic': 'acq', 'type': 'keyword', 'text': 'merger and acquisition agreement'},
    {'id': 'A2', 'topic': 'acq', 'type': 'keyword', 'text': 'takeover bid tender offer for shares'},
    {'id': 'A3', 'topic': 'acq', 'type': 'paraphrase', 'text': 'firm buys a rival business'},
    {'id': 'A4', 'topic': 'acq', 'type': 'paraphrase', 'text': 'investor group seeks control of company'},
    # crude：原油
    {'id': 'C1', 'topic': 'crude', 'type': 'keyword', 'text': 'crude oil prices and OPEC output'},
    {'id': 'C2', 'topic': 'crude', 'type': 'keyword', 'text': 'oil production in barrels per day'},
    {'id': 'C3', 'topic': 'crude', 'type': 'paraphrase', 'text': 'petroleum supply from the gulf'},
    {'id': 'C4', 'topic': 'crude', 'type': 'paraphrase', 'text': 'energy ministers discuss fuel pricing'},
    # grain：穀物
    {'id': 'G1', 'topic': 'grain', 'type': 'keyword', 'text': 'wheat and corn exports'},
    {'id': 'G2', 'topic': 'grain', 'type': 'keyword', 'text': 'grain shipments to the soviet union'},
    {'id': 'G3', 'topic': 'grain', 'type': 'paraphrase', 'text': 'harvest of cereals by farmers'},
    {'id': 'G4', 'topic': 'grain', 'type': 'paraphrase', 'text': 'crop subsidies and farm aid'},
    # trade：國際貿易
    {'id': 'T1', 'topic': 'trade', 'type': 'keyword', 'text': 'trade deficit with japan'},
    {'id': 'T2', 'topic': 'trade', 'type': 'keyword', 'text': 'tariffs on imports and trade sanctions'},
    {'id': 'T3', 'topic': 'trade', 'type': 'paraphrase', 'text': 'protectionism between nations'},
    {'id': 'T4', 'topic': 'trade', 'type': 'paraphrase', 'text': 'dispute over foreign goods entering markets'},
]

# 網頁搜尋框下方的範例查詢（點一下就搜尋）
DEMO_QUERIES = ['petroleum supply from the gulf', 'crude oil prices and OPEC output',
                'firm buys a rival business', 'harvest of cereals by farmers', 'trade deficit with japan']

# 詞層級分析（Top-5 鄰近詞）預設要看的詞：涵蓋 5 個主題，也包含容易誤解的 profit / loss
NEIGHBOR_WORDS = ['oil', 'petroleum', 'opec', 'wheat', 'corn', 'merger', 'takeover',
                  'profit', 'loss', 'dividend', 'tariff', 'deficit', 'japan']
