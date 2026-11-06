# Project 4 進度報告講稿（約 1 分鐘）

老師好，我報告 Project 4 目前的進度。

**資料**：我用 UCI 的 Reuters-21578，照官方 README 的 ModApte 切分。訓練集 9,603 篇拿來訓練 Word2Vec，測試集 3,299 篇拿來做檢索，主題是作業建議的 earn、acq、crude、grain、trade。

**目前完成的部分**：
- Word2Vec 用 gensim 訓練了 CBOW 和 Skip-Gram，另外自己用 numpy 寫了一個小型版本，來理解它們的訓練流程。
- 詞層級分析已經能列出每個詞的 Top-5 鄰近詞。例如 profit 最近的竟然是 loss，說明向量相近不代表是同義詞。
- 檢索的部分，做了方法 A 的 TF-IDF，和方法 B 的 TF-IDF 加權詞向量，用 20 個查詢、以新聞的 topics 當相關性，算 P@10、MRR 和 nDCG。

**初步結果**：
- 用典型關鍵字的查詢，兩種方法差不多。
- 但如果查詢換個說法，例如用 petroleum 找原油新聞，TF-IDF 的 nDCG 只有 0.52，詞向量可以到 0.85。
- 不過也有例外，像「firm reports bigger losses」這題，反而是 TF-IDF 比較好。

**接下來**：我會深入分析這些例外，也會檢查「用主題當相關性」會不會本來就比較偏向詞向量，最後整理成報告。

以上，謝謝老師。
