# =============================================================================
# reuters/queries.py
# Project 4：評估查詢集（設定檔，要新增或修改查詢直接改這裡）
#
# 相關性判斷（作業指定）：以 Reuters 的 topics 欄位為準
#   新聞的 topics 含有查詢對應的主題 → 相關（1），否則不相關（0）
#
# Reuters-21578 是文字分類資料集，沒有附官方查詢，所以這 20 個查詢是自己設計的：
# 每個主題 4 個查詢，分兩種類型：
#   keyword    用該主題新聞最常見、最有代表性的詞（TF-IDF 的主場）；
#              這些詞可以對照 run_reuters_experiments 產生的 topic_keywords.csv（訓練集統計）
#   paraphrase 刻意換一種說法，避開最典型的關鍵字（例如用 petroleum / energy 代替 oil），
#              用來測試 Word Embedding 能不能補上「字不一樣就比對不到」的弱點
# 網頁的搜尋區會把 20 個查詢全部列出來，點一下就搜尋
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

# 網頁一打開時的預設查詢（C3：TF-IDF 與詞向量差異最明顯的例子之一）
DEFAULT_QUERY = 'petroleum supply from the gulf'

# 詞層級分析（Top-5 鄰近詞）預設要看的詞：涵蓋 5 個主題，也包含容易誤解的 profit / loss
NEIGHBOR_WORDS = ['oil', 'petroleum', 'opec', 'wheat', 'corn', 'merger', 'takeover',
                  'profit', 'loss', 'dividend', 'tariff', 'deficit', 'japan']

# 詞層級分析的「試試這些詞」按鈕：每組是一種常見的現象（以 ModApte 模型觀察；hint 是滑鼠提示）
WORD_SUGGESTIONS = [
    ('反義詞也很近', [('profit', '最近的詞是 loss：都出現在 net ___ 的位置'),
                     ('rise', 'fall、drop、decline 都很近'), ('higher', '最近的是 lower'),
                     ('surplus', '最近的是 deficit（順差 vs 逆差）'), ('buy', 'sell 也很近')]),
    ('同一類的詞', [('january', '其他月份'), ('monday', '其他星期'), ('yen', '其他貨幣：marks、guilders、dollar'),
                    ('gold', 'silver、platinum、bullion')]),
    ('人物與組織', [('opec', 'lukman、nazer：1987 年的石油部長'), ('reagan', 'administration、congress'),
                    ('coffee', 'ico：國際咖啡組織')]),
    ('1987 年的新聞事件', [('japan', 'microchips：美日半導體貿易爭端'), ('iran', 'missiles、silkworm：兩伊戰爭'),
                         ('strike', 'seamen、miners：罷工新聞')]),
    ('固定詞組與財報縮寫', [('wall', 'street 的 cosine 高達 0.94'), ('palm', 'rbd、deodorised：棕櫚油規格'),
                          ('shr', 'oper、vs、div：財報表格的縮寫')]),
]
