# 演算法與資料處理規則

本文件對應目前程式實作。字數、句數、字元數的程式 `search/text_processing.py` 與上一個交付版本逐位元相同，本次沒有更動統計規則。三種統計都不會先移除 stopwords 或做 stemming。

## 1. 文章與摘要擷取

索引及分析的範圍是整理後的 **Abstract + Keywords**。不將 bibliographic title、正文、參考文獻或摘要小標題計入；文章標題仍會顯示。PMC 正文可另存 raw_text，當前搜尋不使用正文。

PubMed XML 的各 AbstractText 內容依順序串接，區塊間保留空行；Label 屬性不當作摘要文字。關鍵字來自 MedlineCitation 內 KeywordList。PMC/JATS 以 article-meta 的摘要為主：優先無 abstract-type 的摘要，否則找不是 toc、editor、graphical、teaser、short、lay、plain-language-summary、lay-summary、layperson 的摘要。小標題不計入摘要正文，段落文字保留段落邊界。

碰到 `Keyword:` 或 `Keywords:`（忽略大小寫）之後，不繼續收錄後方其他正文。已有關鍵字行就不重複附加 metadata keywords；否則附加一行 `Keywords: ...`。XML 內的 HTML 實體由 XML parser 還原；統計對象是擷取後文字，不是 XML 標籤或原始檔案位元組。

## 2. Characters 字元數

1. 將所有連續空白（空格、換行、tab 等 Unicode whitespace）換成一個普通空格。
2. 移除首尾空白。
3. `Characters = len(visible_text)`：保留標點和字間空格，Python Unicode code-point 數，不是 UTF-8 bytes 數。
4. `Characters without spaces = len(visible_text.replace(' ', ''))`。

不做 Unicode NFC/NFKC 正規化；組合字與 emoji 的 code-point 數不一定等於肉眼看到的字形數。

## 3. Words 字數與 token 邊界

字數是 `len(tokenize(visible_text))`，不是用空白 split 的數量。token 轉 Unicode casefold，但計數保留 stopword，不做詞幹化；純獨立標點不計 word。

| 規則／例子 | 結果 |
|---|---|
| GLP-1、COVID-19、SARS-CoV-2、IL-6 | 各一個 token |
| 0.05、48.1%、24.9% | 各一個 token |
| 12-15%、12–15% | `12`, `15%`，共二個 |
| 5-stage、age‐specific、non–small | 各一個 token |
| patient's、O’Malley | 各一個 token |
| e.g.、i.e.、U.S. | 各一個 token |
| activity/exercise、Brca1ΔC/ΔC | slash 分開，二個 |
| factors—particularly | em dash 分開，二個 |
| c.1813dupA | 點分開，二個 |
| Δ、β、α、53BP1 | Unicode 字母與數字可成 token |

連接字元允許 `-`、`‐`、`‑`、`–`，需在字母／數字之間；若兩邊都是數字，視為數值範圍分隔。Em dash `—` 不當連接字。底線不是此 tokenizer 的文字連接符。完整 regex 是 `TOKEN_PATTERN`，不要另外用另一套 regex 重算 UI 統計。

## 4. Sentences 句數

使用 `split_sentences(raw_text)`：**先按一個或多個換行分段**，忽略空段落，再在段落內找句尾；不是先把所有換行壓平才斷句。

句尾候選為 `. ! ? 。 ！ ？`。標點後若有結尾引號或括號，會一起納入目前句子；後面必須是空白或段落結尾才切句。不要求下一句是大寫，避免漏掉以 p53、基因名、符號或小寫技術詞開頭的句子。段落最後沒有句尾標點的剩餘文字也算一句。空摘要為零句。

小數點（兩側為數字）、既定縮寫與大寫單字母姓名縮寫 `J. Smith` 不當一般句尾。字數／句數平均值為 `round(words / sentences, 2)`，零句時為零。

既定縮寫清單：

`al, approx, cf, co, dr, e.g, eq, eqs, et, etc, fig, figs, i.e, inc, jr, ltd, mr, mrs, ms, no, nos, p, pp, prof, sr, st, vol, vols, vs`

多點縮寫特別規則：`a.m., e.g., i.e., p.m.`。這是目前規則，不是通用自然語言模型；例如每個換行都可能造成新句。

## 5. Stopword 規則

使用本專案自行定義的固定集合，共 **72** 個。不是下載 NLTK 的英文 stopword corpus。先 tokenize + casefold，再做 **token 完全相等比對**；不做 substring 移除，也不額外刪醫學詞。否定詞 `no`、`not` 在這個既有清單內，維持原規則。

完整清單也附於 `STOPWORDS.txt`：

`a, about, above, after, against, an, and, are, as, at, be, been, before, being, below, between, but, by, can, could, did, do, does, doing, down, during, else, for, from, had, has, have, having, if, in, into, is, it, its, may, might, must, no, not, of, on, or, our, shall, should, so, such, than, that, the, their, then, these, this, those, through, to, up, was, we, were, which, who, whom, will, with, would`

## 6. Porter 與 inverted index

索引流程：擷取 Abstract + Keywords → tokenize/casefold → 移除固定 stopwords → NLTK `PorterStemmer()` 預設模式（NLTK_EXTENSIONS）。不是 lemmatization，也不使用 LLM。詞幹可能不是完整單字，例如 patients → patient、improved → improv。

統計每個 stem 在文章中的 TF，建立 `Term(word)` 和 `Posting(document, term, term_freq)`。索引儲存的 term 最長 100 字元；截斷後聚合 TF。索引批次新增 postings，文章重建時維持 topic memberships。顯示用字數不會使用這份移除停用詞後的索引。

## 7. 搜尋與排名

一般關鍵字與索引使用相同 preprocessing，查詢的重複 stem 只用一次。至少命中一個索引查詢詞就可列入候選（OR）；將各詞的 BM25 貢獻相加，依分數降序。`k1=1.2`, `b=0.75`，doc length 及平均 length 使用處理後 Abstract + Keywords token 數。

BM25 每詞分數：

`ln(1 + (N - DF + 0.5)/(DF + 0.5)) × TF × (k1+1) / (TF + k1 × (1-b + b × doc_len/avg_doc_len))`

N、DF、avg_doc_len 都在目前 topic／年份範圍內計算。此 BM25 的平滑 IDF 和 Zipf 表的 `ln(N/DF)` 不相同。一般搜尋分數顯示四位小數；排序先用原始分數。

若查詢只有停用詞而 preprocess 變成空列表，使用原始 tokenize 的 fallback：需每個查詢 token 都出現（AND），依命中數排序。Zipf 點擊詞彙的查詢則依該 A–D 條件做 exact token 匹配，依次數排序；不是一般 BM25 或 substring。

## 8. Topic／分頁

主題名稱去首尾空白、壓縮內部空白並 casefold 作為識別 key。零個主題＝全部含未分類；一個＝該主題；多個＝聯集，文章去重一次。Search、Articles、Zipf 共用同一個範圍。主題勾選後 450 ms debounce 自動提交，保留搜尋文字／年份／排序／分析條件，重設分頁。

Zipf 的排序、詞彙查詢、範圍及 chart/fit 狀態會保留；圖表、回歸、統計、詞表和匯出重新使用選取後的 corpus。Term sort 是詞表排序，圖表始終使用 CF rank，不會因為改 IDF 排序而把 Zipf 橫軸變成 IDF 順序。

Articles 與 Search results 一頁最多 20 筆；term table 一頁 50 筆，分成兩欄各 25 筆；匯出包含全部符合範圍資料。

## 9. Zipf A–D、CF / DF / IDF

四種條件累積：A＝HW1 word tokens 加獨立標點符號並 casefold；B＝排除完全不含 alphanumeric 字元的 token，保留 GLP-1／小數等完整 token。因此 A/B 會有明確差異。C＝B 去固定 stopwords；D＝C Porter stemming。

只使用非空白摘要；N 是不同摘要文件數。CF 為跨文件出現次數；DF 為含該 term 的不同文件數；IDF = `ln(N/DF)`，沒有 smoothing；coverage = `100*DF/N`。依 CF 降序、同頻 term 字母序決定 rank，從 1 開始。

以所有 vocabulary entries 做不加權 OLS：`ln(CF) = intercept + slope*ln(rank)`；Zipf exponent k = `-slope`。R² = `1-SSE/SST`；RMSE = `sqrt(SSE/V)`，在 natural-log frequency 空間。少於二個詞無擬合；CF 全相同時 R² 未定義。

Head 前 5%、Middle 5–50%、Tail 最後 50%，切點使用 ceil 並以實際 vocabulary 長度截斷。各區另外算 local fit；global RMSE 用整體擬合在該區的誤差。高 R² 本身不證明符合 Zipf's Law。

圖表保留前 50 詞、端點及 log-spaced 點以維持流暢；擬合與 CSV 匯出用所有詞，不只畫出的點。cache key 包含排序後 document IDs 與摘要內容 SHA-256，5 分鐘有效；新文章或不同內容會得到不同 cache key，並非單用 topic 名稱。

## 10. 匯入、重複、失敗處理

主題匯入一次要求 1–1,000 篇，PubMed relevance sort，查詢加上 `hasabstract[text]` 與 `english[lang]`，每次搜尋最多 200 個 ID、每批抓最多 50 個摘要。掃描前 10,000 個搜尋結果；不同 query 的文章不足時可 partial 完成。

先比對 PMCID，再 PMID，再 DOI（忽略大小寫）；沒有穩定 ID 時才用 source filename、title/year fallback。已在同主題就略過；在其他主題則沿用文件並增加 topic link。重複提交數量表示新增這麼多篇，不是指定累計總數。

下載保持 HTTPS certificate 與 hostname 驗證，使用系統信任與 certs/ 的可信 CA。每請求間隔至少 0.36 秒，暫時 HTTP 429/500/502/503/504 或網路錯誤最多四次請求，以 1/2/4 秒等待。批次層再對明確暫時錯誤等待 3/8/20 秒、最多再試三次，畫面顯示原因；等待時仍可停止。SSL 信任錯誤不當暫時錯誤繞過。

搜尋 JSON 或整批 XML 截斷會當成暫時錯誤重試。合法的空 PubMed 批次或只有書籍的批次可略過，繼續找下一批。無摘要或單篇 XML 無法解析：計入 skipped，繼續找下一篇。不可復原的資料庫／檔案錯誤仍回報失敗，不能把真正儲存錯誤當成功。

SQLite 用 WAL + IMMEDIATE、30 秒 busy timeout；正常進度 polling 不寫資料。短暫 busy/locked 只在完整交易回滾後重試，最多四次。文件、索引、topic link、計數同交易提交；失敗清理未提交 XML，不重複計數。只允許一個 active 主題匯入；停止／失敗／容器重啟保留已提交文章。

## 11. 匯出與資料保留

分析 ZIP 包含四種條件完整 term tables、summary、區段指標、corpus 的 IDs／主題／摘要以及 methodology.json。CSV 文字如可被試算表解讀成公式，前置 apostrophe；程式分析文字時依 methodology.json 說明去除。

Docker 的資料庫／XML／logs 放在同一持久化 volume。更新同一資料夾重建映像不清除 volume；不可用 `docker compose down -v` 當一般停止。WAL 運作中不要只複製主 SQLite 檔；用 SQLite backup API 或完整停止工作後備份。

診斷指令：`docker compose exec web python manage.py import_diagnostics`，顯示版本、實際 WAL 設定、最近工作狀態與 log，方便區分資料庫、網路及 XML 錯誤。

## 12. 程式對照

| 規則 | 程式 |
|---|---|
| 統計、token、stopword、Porter、BM25 | search/text_processing.py |
| XML 摘要、metadata、dedup、postings | search/indexer.py |
| 詞頻、A–D、回歸、分區、cache | search/analysis.py |
| 主題聯集 | search/topic_filters.py |
| 匯入、原子提交、batch retry | search/topic_import.py |
| 官方 API／SSL | search/pmc_client.py |
| SQLite bounded retry | search/db_retry.py |
| 搜尋／分頁／匯出 | search/views.py |
| 即時主題、chart preferences | search/static/search/site.js、zipf.js |


## 13. Word2Vec（獨立於 HW1 統計與 BM25）

真正使用 Gensim 4.4.0 Word2Vec。對選取的不同文件，依原 HW1 split_sentences 分句，再以 B/C/D pipeline 處理每一句；不跨句或跨文件建立 context，少於 2 tokens 的句子不參與訓練。B 保留 stopwords，C 去掉同一份72詞 stopwords，D再做Porter。

預設 Skip-gram，亦可選 CBOW；dimensions 50/100（預設100）、window 2/5/10（預設5）、min_count 1/2/5（預設2）、epochs 10/20/40（預設20）。其餘明確固定 negative=5、hs=0、sample=0.001、seed=42、workers=1。初始化 hash 使用 SHA-256 的前8 bytes整數，避免 Python process hash 的隨機化。Gensim預設動態縮短window；初始learning rate 0.025、min_alpha 0.0001。訓練以符合min_count的詞建立詞彙，SG從center預測context，CBOW從context預測center。

只有按 Train 才訓練。模型識別 SHA-256 包含原始abstract內容、文件ID與所有可調參數。Topic切換、文章新增／編輯或參數變更會選取新模型；舊模型不冒充新scope的結果。模型原子寫入 data/embeddings/，與Docker data volume一起保留。不同scope可保留各自模型；不用每次切回就重練。同軟體／硬體環境中穩定hash、單worker與seed支援可重現，不保證跨版本bitwise一致。

相似詞是模型原始向量的cosine similarity = dot(u,v)/(norm(u)norm(v))，排除query本身、取最多10詞。查詢先走同一condition處理，需剛好一個token；OOV明確告知，不生成假結果。PCA圖只對query與這10個neighbors的向量置中後做SVD，取前兩個主成分；保留variance ratio。圖上距離近不等於cosine分數完全相同；不同模型向量軸可旋轉，不直接逐座標比較。Export含標準文字Word2Vec向量（第一列vocabulary dimensions）、metadata、corpus local IDs、當前相似詞與projection。

模型訓練完成後另對依模型詞頻排序的前40詞做整體PCA：向量矩陣以各維平均值置中後SVD，取前兩軸；點大小使用該詞的corpus count。這張圖用於概覽，與query加10個neighbors的局部PCA分開計算，不能以二維距離取代原始維度的cosine ranking。Export另含`vocabulary_pca.json`。

## 14. Optional跨領域比較

Domain A/B各自選一或多個topic，組內union distinct。預設交集文件同時從两邊移除，亦可include。Numeric sample預設500、另可100：先去交集，再取兩邊較小的可用篇數（最多cap），兩邊都依local ID升序取前N。這是確定性抽樣，不是隨機或日期配對；只控制篇數，未控制發表年份。All則各自用所有符合文件，篇數可以不同。無主題不是全部，該domain為空；不足時顯示明確提示。

每邊用同一condition與同一Zipf/CF/DF算法；也匯出全A–D，保留exact IDs與摘要。Top10按CF、再字母排序；Jaccard=|Top10A∩Top10B|/|Top10A∪Top10B|。不同來源、時間、篇數、keywords、abstract style會影響差異；不把描述性差異當成domain的因果效應。

## 15. CS来源与identifier

arXiv Atom API固定限制cat:cs.*，查詢可用cat:cs.IR、cat:cs.LG、cat:cs.AI或keyword；最多100 records/request、單連線、每次（包括request retries）至少3.1秒。只取summary（abstract）與bibliographic metadata，不取PDF/body。多數arXiv摘要是英文，但API沒有PubMed式english filter，不自動聲稱已驗證所有語言。需自行檢查樣本。

arXiv ID去掉v1/v2等version尾碼，當同篇版本去重。Adapter XML保留原abstract作AbstractText、另存ArticleId IdType=arxiv；PMID欄位保持空，不捏造PMID。分類不附加成Keywords，避免增加摘要外的category tokens；與原HW1 PubMed/PMC摘要可能附Keywords的差異需在報告交代。去重增加arXiv ID檢查，仍保留PMCID/PMID/DOI與原fallback；index/statistics沿用同一套HW1函式。

## 16. Unicode、編輯距離與大小寫

UTF-8是bytes→Unicode text的解碼；XML以其宣告由ElementTree解碼。Unicode正規化是另一階段：NFC合併canonical equivalents（例如é與e+combining acute）；NFKC還合併compatibility forms（例如全形Ａ）。casefold較ASCII lowercase更完整，例如Straße→strasse。Text matching提供none/NFC/NFKC及case-sensitive/casefold。

Levenshtein 以 Unicode codepoints 為單位，插入、刪除、替換成本 1、相同成本 0。Text matching 輸入單一詞後，對 Condition B 完整 vocabulary 使用有界 DP 找出距離 1–4 的全部候選，依距離分組並按 DF、CF 排序；選取候選時顯示完整 DP 矩陣及一條最佳路徑。不是 Damerau distance：ab→ba 需 2。不是 UTF-8 byte 或 grapheme cluster distance；只作 spelling lookup，Word2Vec 才是 context-based similarity。

HW1 document_stats/tokenization仍完全不變，Text matching的normalization選項不偷偷套用到原搜尋index。原BM25與Zipf使用既有Unicode-aware tokens和casefold；不宣稱已具NFC/NFKC索引。若未來導入正規化index，必須另版明確重建索引並保留原顯示統計。

## 17. 報告與TF-IDF

Report以當下topic union与所選condition重新計算，對RQ1–RQ5、R²限制與head/middle/tail提供數值連結的草稿；不以R²門檻自動宣告power law成立。最佳region按global fitted-line RMSE最小選，至少2terms。CF/DF示例從當前資料選，不用虛構教科書數值。IR discussion固定300–500 English words，畫面顯示split()詞數；學生仍需核對、加入課堂引用與自己的interpretation。

另算per-document TF-IDF：TF為該condition的原始occurrence count，IDF=ln(N/DF)，乘積TF×IDF。空abstract排除N；DF文件只算一次。與BM25的IDF不同，沒有改搜尋ranking。完整tfidf.csv包含所有document-term非零TF組合。CSV公式開頭字串以前置apostrophe防spreadsheet公式執行。Executive Summary可開專屬列印view，A4樣式排成一頁，包含四個findings；亦匯出Markdown，不把draft當已完成學生個人報告。

## 官方參考

- Gensim Word2Vec: https://radimrehurek.com/gensim/models/word2vec.html
- arXiv API: https://info.arxiv.org/help/api/user-manual.html
- arXiv API terms/rate limits: https://info.arxiv.org/help/api/tou.html
- Unicode UAX #15: https://www.unicode.org/reports/tr15/


### 本版標點規則（依使用者最新要求）

獨立的 . , ; 等標點在 Zipf Condition A 中各自作為 term；Condition B 明確排除獨立標點，因此 A/B 不同。詞內 GLP-1、COVID-19、apostrophe、decimal 與 percent 仍沿用 HW1 完整 token，不刪掉生醫詞連接。不改 HW1 字數、句數及字元算法。


CS CSV備用：arXiv API可能回429限流。可上傳本機UTF-8 CSV（headers id,title,abstract；id保留真實arXiv ID），最多1,000rows/20MB。版本尾碼去重、單列invalid略過並提示、不捏造PMID、不抓正文，仍沿用HW1摘要索引及統計。API429則有界重試後顯示真正原因，已完成文章保留。


## 18. 搜尋欄拼字建議（2026-10-04）

正常 BM25 搜尋完成後，在搜尋欄下方提供 Did you mean?，不取代原始查詢或原結果。零筆結果或已有部分匹配但查詢包含疑似拼錯的詞，都可能出現建議；沒有足夠接近的候選則不顯示。點選候選才執行新搜尋，保留 topic union 與年份，重設頁碼。主題改變後，結果與建議一起重新計算。Zipf exact-term 查詢不產生拼字替換。

連字號只作查詢時的等價展開，不改寫儲存的 token 或 HW1 統計。例如 `glp1` 與 `GLP-1`、`covid19` 與 `COVID-19` 會合併同一組 postings，以聯集 DF 及每篇合計 TF 計算一次 BM25；畫面會列出額外納入的變體。這是確定性的標點正規化，因此自動套用；`camcer` 之類的編輯距離猜測仍只顯示 Did you mean?，不自動改查詢。apostrophe、句點及純數字不折疊，以避免 `cant`/`can't` 或數值範圍誤配。

候選字典只來自當前主題與年份範圍的 Abstract + Keywords（沿用 Document.abstract），使用原 HW1 tokenize，另為建議做 NFC + casefold；不改原始文章字数、句數、字元數、索引或 BM25。已存在的詞、Porter 詞幹已存在的詞、72 個 stopwords、數字、含數字/連字符的 biomedical identifiers、少於 4 字元與超過 32 字元的詞均不建議替換。未知詞不是必然錯字，畫面只提供可能的修正。

候選使用有界 Levenshtein（Unicode codepoints；插入、刪除、替換各 1；相鄰交換需 2），搜尋建議允許距離 1–3。先依編輯距離升序，再依當前範圍 DF 降序、CF 降序、字母順序排序。多個疑似錯詞使用寬度 3 的 beam 合成完整查詢；總距離最小優先，再依 DF/CF 加總排序，最多 3 個不同查詢，且每個建議顯示總編輯距離。保留查詢中未替換的詞、標點、空格，以及替換詞原有的全大寫/首字大寫形式。

查詢長度超過 512 字元或 12 個 tokens 時跳過建議，限制計算成本。字典快取 300 秒，以範圍內全部文章 ID 與 abstract 的 SHA-256 為鍵；文章新增、修改或範圍改變會使用不同快取鍵。這是拼字修正，不是 Word2Vec 語意推薦；Word2Vec 保留於原專用分析頁。
