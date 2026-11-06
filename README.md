# BioMed IR：生醫資訊擷取課程網站（HW1 → Project 2 → Project 3）

這是用 Django 寫的生醫文獻全文檢索網站，用 Docker 執行。歷次作業都做在同一個網站裡。

| 作業 | 內容 | 網頁位置 |
|---|---|---|
| HW1 | XML 解析、tokenization、句子切分、BM25 搜尋、文章匯入 | 搜尋、文章、匯入 |
| Project 2 | Zipf 分析（A–D 前處理）、比較領域、Word2Vec、文字比對 | 分析 → Zipf / 比較領域 / Word2Vec / 文字比對 |
| Project 3 | Term Weighting、向量空間檢索、對照實驗、評估 | 分析 → 詞項權重（`/analysis/term-weighting/`） |
| **Project 4** | **Reuters-21578：TF-IDF vs Word2Vec 新聞檢索** | **主選單 → Reuters**（`/reuters/`） |

---

## 1. 啟動

需要先安裝並開啟 Docker Desktop（Linux containers）。在有 `docker-compose.yml` 的資料夾開 PowerShell：

```powershell
# 第一次啟動，或改過程式碼之後（會重新 build）
docker compose up --build

# 之後一般啟動
docker compose up
```

瀏覽器打開 **http://127.0.0.1:8001/**。按 Ctrl+C 停止。

- 啟動時會自動套用資料庫 migration，既有文章都會保留。
- 如果學校或公司網路有 HTTPS 憑證問題（無法匯入文章），先執行一次 `powershell -ExecutionPolicy Bypass -File .\prepare_certs.ps1` 再 build。細節見 [docs/hw1_hw2_notes.md](docs/hw1_hw2_notes.md)。
- 如果 8001 port 被占用：複製 `.env.example` 成 `.env`，把 `BIOMEDIR_PORT` 改成別的 port。

## 2. Project 3：Term Weighting

### 網頁

打開「分析 → 詞項權重」。整個作業都在這一頁，由上到下對應作業的五個任務：

| 段落 | 對應任務 | 內容 |
|---|---|---|
| 控制變因設定 | 任務一 | stopword removal、Porter stemming 開關（整頁共用） |
| ① 詞項統計 | 任務一 | 文件數、token 數、詞彙量、文件長度分布、DF 前 20 名、查單一詞的 TF/DF/CF/IDF |
| ② 權重方法 | 任務二 | Raw TF、Log TF、IDF、TF-IDF、Log TF-IDF（必要）＋ BM25（額外比較）的公式 |
| ③ 搜尋 · Top 10 | 任務三 | 用下拉選單選權重方法與 L2 開關；顯示排名、PMID、標題、分數、摘要片段（命中詞高亮） |
| ④ 比較模式 | 任務三 | 同一查詢、六種方法的 Top 10 並排，▲▼ 標示相對於所選方法的排名變化 |
| ⑤ 評估 | 任務五 | 指標說明（可展開）、12 種組合的平均指標表、逐查詢 nDCG@10 熱圖表、指標總覽熱圖、MRR 的 L2 前後比較圖 |

任務四（對照實驗：RQ1 Raw TF vs Log TF、RQ2 高 / 低 DF 詞的 IDF、RQ4 長短文件與 L2）的表格不放在網頁上，由實驗指令輸出成 CSV / PNG，並寫在報告裡。網頁上的程式碼只是註解掉，要恢復請看 `weighting/templates/weighting/term_weighting.html` 裡的說明。

### 一鍵重跑全部實驗（產生 CSV 與 PNG）

```powershell
docker compose exec web python manage.py run_weighting_experiments
```

結果會寫到 `docs/project3_results/`。這個資料夾有掛載到容器，所以本機也看得到。報告與 Demo 腳本：

- [docs/project3_report.md](docs/project3_report.md)：系統說明、公式、實驗設計、結果、RQ1–RQ5、限制
- [docs/project3_demo.md](docs/project3_demo.md)：Demo 流程

### 修改評估查詢

查詢與相關性規則（MeSH）都寫在 [weighting/queries.py](weighting/queries.py)，改完重跑上面的指令即可。

## 2b. Project 4：Reuters-21578 新聞檢索

打開上方主選單的「Reuters」（中文介面顯示「Reuters 新聞」，網址 `/reuters/`）。由上到下：① 資料集（README 的三種切分，預設 ModApte）② Word2Vec 訓練（gensim CBOW / Skip-Gram ＋ 自製 numpy 示範）③ 詞層級分析（Top-5 鄰近詞、PCA / t-SNE）④ 新聞搜尋（方法 A / B 並排）⑤ 文件層級評估（P@10、MRR、nDCG@10，以及三種切分的比較）。

```powershell
# 一鍵重跑：第一次會從 UCI 下載 Reuters-21578，三種切分各訓練一組 Word2Vec（約 15 分鐘），之後讀快取
docker compose exec web python manage.py run_reuters_experiments
# 只重跑某個切分
docker compose exec web python manage.py run_reuters_experiments --split modapte
docker compose exec web python manage.py test reuters
```

- 資料：UCI Machine Learning Repository 的 Reuters-21578，存在 `data/reuters/`（不進 git）
- 切分（官方 README 第 VIII 節）：ModApte（預設，README 推薦、作業指定）、ModLewis、ModHayes；在 ① 切換後整頁都改用該切分
- 評估查詢（20 個，搜尋區全部列出可直接點）與 topic 相關性規則：[reuters/queries.py](reuters/queries.py)
- 實驗輸出：`docs/project4_results/<切分>/`；三種切分共用的比較表、numpy 示範圖在 `docs/project4_results/` 最上層
- 第一次開頁面（或第一次切換到某個切分）需要 10–20 秒載入模型，之後很快；Demo 前先開一次
- 頁面完整說明（由上到下）：[docs/project4_page_guide.md](docs/project4_page_guide.md)；一分鐘進度報告：[docs/project4_progress_speech.md](docs/project4_progress_speech.md)
- 新聞列表 `/reuters/news/`：資料集名稱 Reuters-21578，依 TOPICS 主題（earn、acq、crude…）瀏覽全部 21,578 篇；「文章」頁上方也有入口
- 新聞內頁 `/reuters/news/<NEWID>/`：全文、主題、所屬切分、查詢詞高亮、這篇在各方法的排名

## 3. 專案結構

```
bir_main_project/
├─ README.md                  ← 這份說明
├─ docker-compose.yml         Docker 執行設定（port、資料夾掛載）
├─ Dockerfile                 Docker 映像檔（Python 3.12 + requirements.txt）
├─ docker-entrypoint.sh       容器啟動時：建立資料夾 → migrate → 檢查資料
├─ requirements.txt           Python 套件
├─ manage.py                  Django 指令入口
├─ prepare_certs.ps1          匯出 Windows 信任憑證給 Docker 用（網路有憑證問題時）
│
├─ biomedir/                  Django 專案設定（settings.py、最上層 urls.py）
│
├─ search/                    HW1 + Project 2 的所有功能
│  ├─ models.py               資料表：Topic、Document、Term/Posting、ImportJob
│  ├─ views.py                搜尋、文章、Zipf、匯入頁面
│  ├─ research_views.py       比較領域、Word2Vec、報告、文字比對頁面
│  ├─ text_processing.py      HW1 tokenizer、stopword、stemming、BM25（Project 3 也共用）
│  ├─ indexer.py              解析 PubMed / PMC XML、建立 HW1 倒排索引
│  ├─ analysis.py             Zipf A–D 分析
│  ├─ embeddings.py           Word2Vec
│  ├─ spelling.py             拼字建議 + 編輯距離
│  ├─ importing/              從 NCBI / arXiv / CSV 匯入文章
│  ├─ tests/                  search 的測試
│  ├─ templates/search/       頁面 HTML
│  └─ static/search/          CSS、JavaScript（含中英文翻譯 i18n.js）
│
├─ weighting/                 ★ Project 3（Term Weighting）全部在這裡
│  ├─ preprocessing.py        任務一：可開關的前處理
│  ├─ schemes.py              任務二：權重公式、L2、cosine（numpy / scipy.sparse）
│  ├─ index.py                任務一＋三：建立並快取稀疏矩陣、詞項統計、搜尋排名
│  ├─ queries.py              任務五：評估查詢與 MeSH 相關性規則（設定檔）
│  ├─ evaluation.py           任務五：相關性判斷與 P@k、Recall@k、MRR、nDCG
│  ├─ experiments.py          任務四＋五：對照實驗表格、CSV 與 PNG 輸出
│  ├─ views.py / urls.py      網頁
│  ├─ templates/weighting/    頁面 HTML
│  ├─ static/weighting/       頁面 CSS
│  ├─ management/commands/run_weighting_experiments.py   一鍵重跑實驗
│  └─ tests.py                4 篇小語料手算驗證
│
├─ reuters/                   ★ Project 4（Reuters-21578 新聞檢索）全部在這裡
│  ├─ corpus.py               下載 UCI 資料、安全解壓、解析 SGML、ModApte 切分
│  ├─ text.py                 前處理（TF-IDF 與 Word2Vec 共用）、切句
│  ├─ embedding.py            gensim 訓練 CBOW / Skip-Gram、鄰近詞
│  ├─ toy_word2vec.py         自己用 numpy 實作的 CBOW / Skip-Gram（教學示範）
│  ├─ static/reuters/projection.js  可縮放的「誰跟誰比較近」互動散布圖
│  ├─ retrieval.py            方法 A：TF-IDF；方法 B：TF-IDF 加權詞向量（＋ C、D 額外比較）
│  ├─ queries.py              20 個評估查詢（keyword / paraphrase）
│  ├─ evaluation.py           topic 相關性與 P@10、MRR、nDCG@10
│  ├─ experiments.py          所有 CSV / PNG 輸出
│  ├─ views.py / urls.py / templates/ / static/
│  ├─ management/commands/run_reuters_experiments.py
│  └─ tests.py
│
├─ docs/                      文件
│  ├─ project3_report.md      Project 3 報告
│  ├─ project3_demo.md        Project 3 Demo 腳本
│  ├─ project3_results/       Project 3 實驗輸出（CSV、PNG）
│  ├─ project4_results/       Project 4 實驗輸出（CSV、PNG、互動圖座標 projection.json）
│  ├─ project4_page_guide.md  Project 4 頁面完整說明
│  ├─ project4_progress_speech.md  Project 4 一分鐘進度報告
│  ├─ project2_checklist.md   Project 2 需求對照
│  ├─ algorithm_rules.md      HW1 / Project 2 演算法規則（tokenizer、句子切分、Zipf…）
│  ├─ import_guide.md         匯入文章使用說明（中文）
│  ├─ hw1_hw2_notes.md        HW1 / Project 2 詳細操作與更新紀錄（英文，原 README）
│  ├─ hw1_validation.json     HW1 驗證資料
│  └─ stopwords.txt           stopword 清單
│
├─ data/                      執行資料（不進 git，會掛載到容器）
│  ├─ db.sqlite3              資料庫
│  ├─ corpus/                 文章 XML
│  ├─ embedding_cache/        Word2Vec 模型快取
│  └─ retrieval_cache/weighting/   Project 3 稀疏矩陣快取
└─ certs/                     額外信任的 CA 憑證（選用）
```

## 4. 資料

目前資料庫有 3500 篇文章：
- **3000 篇 GLP-1 主題的 PubMed 摘要**：Project 3 使用這組。每篇都有 PMID，XML 裡也都有 MeSH 標註。
- 500 篇 arXiv CS 摘要：Project 2 的比較領域使用。

所有資料都在 `data/`。`docker compose down` 或重新 build 都不會刪除它。要新增文章，請用網頁的「匯入」，說明見 [docs/import_guide.md](docs/import_guide.md)。

**PubMed 匯入只收有 MeSH 標註的文章**（沒有 MeSH 的會被跳過、算在 skipped），因為 Project 3 的相關性判斷依賴 MeSH。補了文章之後，要重跑 `run_weighting_experiments` 才會更新實驗結果。

## 5. 開發與測試

```powershell
# 在 Docker 裡跑全部測試（search + weighting）
docker compose exec web python manage.py test

# 只跑 Project 3 的手算單元測試
docker compose exec web python manage.py test weighting
```

不用 Docker、直接在本機跑時，要先指定資料庫位置：

```powershell
$env:BIOMEDIR_DB = "data/db.sqlite3"
python manage.py runserver
```
