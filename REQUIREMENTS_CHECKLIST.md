# Project 2 需求逐項核對

依據本次提供的 project-2-details-20260929(3).pdf，完整 8 頁；另納入使用者補充的 Word2Vec、編輯距離與 Unicode。

| 要求 | 實作／位置 | 說明 |
|---|---|---|
| 約 1,000 篇英文 abstracts、唯一 PMID | Upload → PubMed；Articles | 每次 1–1,000；真正的 PMID 保留。報告顯示實際篇數與有 PMID 篇數，不假稱已收滿 1,000。 |
| Part I tokenization / case folding / punctuation / stopwords / stemming | Analysis → Zipf A–D；ALGORITHM_RULES.md | 相同摘要，四條累進 pipeline；保留原 HW1 統計算法。 |
| Part II 文件數、tokens、vocabulary、average | Zipf corpus metrics / Compare all | 全部以實際選取的不同文件計算。 |
| Part III CF 排序、Top 50 | Zipf → Term statistics | 一頁兩欄、每欄 25 筆，直接顯示 Top 50；完整 CSV 匯出。 |
| Part IV DF | Zipf term table | 每篇文章至多為每個 term 的 DF 貢獻 1。 |
| Part V Rank–frequency / log–log | Zipf 圖表模式 | 完整 rank 範圍；可儲存 SVG。 |
| Regression slope/intercept/k/R²/RMSE | Zipf regression strip / export | 全 vocabulary ranks 的 natural-log OLS。 |
| Q1–Q4: Zipf、exponent、R² 限制、最佳區段 | Research report；Zipf regional fit | 實際數值＋有界解釋；不把 R² 當 power-law 證明。 |
| Part VI 比較 A/B/C/D vocabulary / leading terms / exponent / shape | Zipf → Compare all；Research report RQ3 | 疊圖與摘要指標；保留 shape 觀察及 residuals。 |
| Part VII 至少 20 terms CF vs DF＋三個問題 | Research report CF/DF evidence & RQ4 | vocabulary 足夠時列出 20；例子取自本次選取集合。 |
| Part VIII 至少 10 terms CF/DF/IDF＋common-term 解釋 | Research report、Zipf表 | 顯示20 terms；另有 per-document TF × IDF 及完整匯出。 |
| Part IX 300–500 English words IR discussion | Research report | 內容涵蓋 stopword / dictionary / postings / storage / compression / TF-IDF；畫面顯示實際詞數。 |
| Optional Compare Two Domains | Analysis → Compare domains | 各自選多主題，Documents/Tokens/Vocabulary/k/R²/Top10，疊圖與匯出；預設排除交集、平衡篇數。 |
| Optional: Does domain specificity affect distribution? | Compare domains observations/export | 依實測描述差異；明列 source、sample size、date/style 的混雜因素，不做因果斷言。 |
| Final challenge empirical approximate law usefulness | Research report RQ5 | 連結當前數據、索引設計、retrieval concepts；仍需學生加入課堂／教科書參考。 |
| One-page Executive Summary 四項 finding | Research report → Open print summary | 可直接 Print/Save PDF；獨立 summary.md 匯出。 |
| Final RQ1–RQ5 | Research report | 全部列出；數值隨目前主題／condition 重算。 |
| 使用者要求：Word2Vec | Analysis → Word2Vec | 真正 Gensim Skip-gram/CBOW；參數、neighbors cosine、PCA、vectors.txt export。 |
| 使用者要求：編輯距離 | Analysis → Text matching | Levenshtein，插入／刪除／替換=1；Unicode code points。 |
| 使用者要求：Unicode／大小寫 | Text matching；ALGORITHM_RULES.md | UTF-8 解碼 vs NFC/NFKC vs casefold 分開說明；不改 HW1 統計。 |
| CS 摘要來源 | Upload → arXiv CS | Atom API，cat:cs.*；保留 arXiv ID，不捏造 PMID；Medical 用 PubMed。 |
| Topic 切換同步 | Search / Articles / Zipf / Word2Vec / report / matching | 自動刷新；Domains 有自己的兩組選取。 |

## 必須由實際實驗與學生完成的部分

程式不替你假造 1,000 篇實驗結果。匯入真實集合後，確認 PMID coverage、英文來源與篇數，下載 exact corpus 與結果。Report 是以實際數值帶入的草稿；請核對曲線、寫下自己看到的轉折、加入課堂和教科書的引用，再作為 final report。Pure CS arXiv 沒有 PMID；主作業可維持 1,000 PubMed abstracts，Optional 的 CS 比較則清楚列 arXiv identifiers。若老師要求 Optional 也每篇有 PMID，先確認是否接受來源原生 ID。


### 本版標點規則（依使用者最新要求）

獨立的 . , ; 等標點在 Condition A 中各自作為 term，Condition B 排除獨立標點，因此 A/B 有明確差異。詞內 GLP-1、COVID-19、apostrophe、decimal 與 percent 仍沿用 HW1 完整 token，不刪掉生醫詞連接。不改 HW1 字數、句數及字元算法。


搜尋拼字建議：`search/spelling.py` + Search 頁；零筆/部分匹配可提示，最多 3 筆，點選才搜尋，保留主題/年份；切換主題同步刷新。保留原結果與 HW1 統計，規則見 ALGORITHM_RULES.md §18。
