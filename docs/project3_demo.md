# Project #3 Demo 流程（約 10 分鐘，3000 篇版本）

完整逐字講稿見 [project3_speech.md](project3_speech.md)；這份是操作順序與重點清單。

**事前準備**
1. 開 Docker Desktop，等到綠燈。
2. `docker compose up -d`，打開 http://127.0.0.1:8001/analysis/term-weighting/，右上角切「繁中」，按 Ctrl+F5 確保是最新版。
3. 另開一個分頁放報告或一頁 Word（`docs/蔡佳彣_VI6152022_hw3.docx`）備用。

---

| 順序 | 畫面操作 | 要講的重點（數字） |
|---|---|---|
| 0 | 頁面最上方 | 3000 篇 GLP-1 PubMed 摘要、每篇都有 MeSH；權重用 numpy / scipy 自己實作 |
| 1 | 控制變因設定：stemming 開 → 套用 → 關 | 詞彙量 25,865 → 約 20,560；整頁會重算 |
| 2 | ① 語料統計 | 42 萬 token、詞彙量 25,865、42% 的詞只出現一次；長度 4–506，平均 143 |
| 3 | 查詢單一詞項：semaglutide | CF 568、DF 247、IDF 1.08 |
| 4 | DF 前 20 名 | glp-1 DF 2955（98.5%）、IDF 0.007；「2」≈ type 2（第二型糖尿病）、「1」≈ glucagon-like peptide 1；glucagon-like 2355 vs peptide-1 1844 |
| 5 | ② 權重方法 | 5 種必要 ＋ BM25；L2 開 = cosine、關 = 內積；共 12 組合 |
| 6 | ③ 搜尋：GLP-1 obesity weight loss、Log TF-IDF、L2 開 | 查詢向量 glp-1 / obesity / weight / loss；第 1 名 50 tokens 的短摘要；黃色高亮 |
| 7 | 改 Raw TF、L2 關 | 分數變 34、33；第 1 名是 glp-1 重複 34 次的動脈硬化論文 → **RQ1** |
| 8 | 改 Log TF-IDF、L2 關 | 前幾名 200–300 tokens（第 1 名 294）；開 L2 只有 50–90 → **RQ4** |
| 9 | 設回 Log TF-IDF、L2 開，④ 比較模式 | 用同一個查詢；Raw TF 只重疊 3 篇、BM25 重疊 9 篇 |
| 10 | ⑤ 評估：點開「這些指標代表什麼？」 | MeSH 規則（主要主題 2 分、其他 1 分）；指標是評估用，不是權重方法 |
| 11 | 平均表 | nDCG 最高 Log TF-IDF 無 L2（0.842）；P@5 / P@10 最高 TF-IDF 無 L2（0.933 / 0.908）；IDF only 最低（0.531）→ **RQ3、RQ5**；Recall@10 上限 0.07 |
| 12 | 逐查詢 nDCG 表 | 每題最佳方法不同；Q05 Raw TF = 0 |
| 13 | 熱圖、MRR 圖 | L2 讓 Log TF、IDF 的 MRR 升到 1，TF-IDF、Log TF-IDF 下降；nDCG 上 L2 只幫到 Raw TF、IDF → **RQ4** |
| 14 | 結尾 | 四個學習重點；1000 篇時 TF-IDF + L2 最好，3000 篇變成 Log TF-IDF 無 L2 最好 → 方法優劣要看資料集 |

（可選）終端機示範重現：

```powershell
docker compose exec web python manage.py run_weighting_experiments   # 印出 12 組合平均表，更新 docs/project3_results/
docker compose exec web python manage.py test weighting              # 22 個手算單元測試，全部 OK
```

---

### 備用問答

| 可能被問 | 回答重點 |
|---|---|
| MRR 為什麼不在權重方法的下拉選單？ | MRR 是評估指標，用來替排名打分數；權重方法才是決定排名的方式 |
| 比較模式用的是哪個查詢？ | 搜尋框裡的同一個查詢與同一個 L2 設定，只換權重方法 |
| 為什麼 log 用 10 為底？ | 依作業規格；換底只會把所有 IDF 乘同一個常數，有 L2 時不影響排名 |
| 沒 L2 也叫 cosine 嗎？ | 不是。沒 L2 是內積，有 L2 才是 cosine；這就是 RQ4 的比較 |
| 為什麼 L2 反而讓很多方法變差？ | L2 後排名看查詢詞密度，很短、剛好提到查詢詞的摘要會排很前面；BM25 本身已有長度正規化，再加 L2 等於重複懲罰 |
| 相關性怎麼來？ | 作業允許用主題標籤；規則在 weighting/queries.py，限制寫在報告第 7 節 |
| 為什麼每篇都有 MeSH？ | 匯入時規定 PubMed 文章一定要有 MeSH，沒有就跳過 |
| 對照實驗（任務四）在哪？ | 實驗指令輸出成 CSV / PNG，寫在報告第 5 節；網頁上的「比較模式」也是控制變因的對照 |
