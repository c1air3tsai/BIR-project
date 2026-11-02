# =============================================================================
# weighting/queries.py
# Project 3 任務五：評估查詢集（設定檔，要新增或修改查詢直接改這裡）
#
# 資料集沒有現成的 relevance labels，所以用 PubMed 文件的 MeSH 標註建立相關性：
#   - 每個查詢列出幾個 MeSH Descriptor（mesh）或 MeSH Qualifier（qualifiers）
#   - 文件的 MeSH 有其中任何一個，且該 heading 是 Major Topic  → relevance = 2（高度相關）
#   - 有其中任何一個，但不是 Major Topic                         → relevance = 1（相關）
#   - 都沒有                                                      → relevance = 0（不相關）
# 規則實作在 weighting/evaluation.py 的 relevance_grades()。
#
# 選 MeSH 的原則：只選語料中實際出現過的 descriptor（行尾註解為目前 3000 篇中的相關篇數），
# 查詢文字用一般使用者會打的字，不直接照抄 MeSH 名稱，以降低「規則與查詢詞重疊」的偏差。
# =============================================================================

EVALUATION_QUERIES = [
    {'id': 'Q01', 'text': 'GLP-1 receptor agonists for obesity and weight loss',
     'mesh': ['Obesity', 'Obesity, Morbid', 'Weight Loss', 'Anti-Obesity Agents']},  # 648 篇
    {'id': 'Q02', 'text': 'semaglutide',
     'mesh': ['Semaglutide']},  # 143 篇
    {'id': 'Q03', 'text': 'liraglutide treatment',
     'mesh': ['Liraglutide']},  # 266 篇
    {'id': 'Q04', 'text': 'cardiovascular outcomes and heart failure',
     'mesh': ['Cardiovascular Diseases', 'Heart Failure', 'Cardiovascular System', 'Myocardium']},  # 148 篇
    {'id': 'Q05', 'text': 'side effects and safety of GLP-1 drugs',
     'qualifiers': ['adverse effects']},  # 293 篇
    {'id': 'Q06', 'text': 'insulin secretion from pancreatic beta cells',
     'mesh': ['Insulin-Secreting Cells', 'Insulin Secretion', 'Islets of Langerhans']},  # 336 篇
    {'id': 'Q07', 'text': 'appetite and food intake regulation',
     'mesh': ['Appetite', 'Appetite Regulation', 'Eating', 'Satiation', 'Satiety Response', 'Feeding Behavior']},  # 283 篇
    {'id': 'Q08', 'text': 'DPP-4 inhibitors',
     'mesh': ['Dipeptidyl-Peptidase IV Inhibitors', 'Dipeptidyl Peptidase 4']},  # 213 篇
    {'id': 'Q09', 'text': 'bariatric surgery gastric bypass',
     'mesh': ['Bariatric Surgery', 'Gastric Bypass', 'Gastrectomy']},  # 117 篇
    {'id': 'Q10', 'text': 'neuroprotection in Alzheimer and Parkinson disease',
     'mesh': ['Alzheimer Disease', 'Parkinson Disease', 'Neurodegenerative Diseases', 'Neuroprotective Agents']},  # 97 篇
    {'id': 'Q11', 'text': 'nonalcoholic fatty liver disease',
     'mesh': ['Non-alcoholic Fatty Liver Disease', 'Fatty Liver']},  # 47 篇
    {'id': 'Q12', 'text': 'gut microbiota',
     'mesh': ['Gastrointestinal Microbiome']},  # 77 篇
]

# 網頁搜尋框下方的範例查詢（點一下就搜尋）
DEMO_QUERIES = [
    'GLP-1 obesity weight loss',
    'semaglutide',
    'side effects nausea',
    'insulin secretion beta cells',
    'DPP-4 inhibitors',
]

# 任務四 RQ1 / RQ2 詞權重示範使用的詞（會經過與文件相同的前處理）
#   高 DF 的常見詞（glp-1、patients、insulin）與低 DF 的專門詞（semaglutide、microbiota、tirzepatide）
DEMO_TERMS = ['glp-1', 'insulin', 'patients', 'obesity', 'semaglutide', 'microbiota', 'tirzepatide']

# 任務四 RQ4 長短文件示範使用的查詢
LENGTH_DEMO_QUERY = 'GLP-1 obesity weight loss'
