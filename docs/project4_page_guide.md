# Project 4「Reuters 新聞檢索」頁面完整說明（由上到下）

> 開啟方式：上方主選單的「**Reuters**」，網址 `http://127.0.0.1:8001/reuters/`（舊網址 `/analysis/reuters/` 會自動轉過來）。
> 第一次開頁面要約 10 秒（把 Word2Vec 模型和 3,299 篇新聞的向量載入記憶體），之後每次不到 1 秒。Demo 前先開一次。

這一頁要回答作業的核心問題：**TF-IDF 只認得「一模一樣的字」，Word2Vec 會從上下文學到詞義；把詞向量加進檢索，排名到底有沒有變好？**

頁面最上方有一排「段落導覽」（1 資料集 … 5 文件層級評估）。它會固定在頁首下方，捲到哪一段，那一段就會變成深色；點一下就跳到該段。手機上這一排可以左右滑動。導覽列最右邊的「切分 · ModApte」提醒你：整頁目前用的是哪一種切分。

在任何一段送出表單（搜尋、查鄰近詞）或點查詢 / 詞的按鈕，頁面重新載入後會**自動回到原本那一段、停在一樣的螢幕位置**，不用再從頁首往下拉。

---

## ① 資料集：Reuters-21578（ModApte 切分）

### 切分選單：README 的三種切分

官方 README 第 VIII 節定義了三種切分，上方三張卡片可以直接切換（網址參數 `split`）。切換後**整頁都換掉**：Word2Vec 改用該切分的訓練集重新訓練的模型、搜尋範圍改成該切分的測試集、評估結果也換成該切分的。

| 切分 | 訓練集 | 測試集 | 未使用 | 規則 |
|---|---|---|---|---|
| **ModApte**（預設） | 9,603 | 3,299 | 8,676 | `LEWISSPLIT` 為 TRAIN / TEST，**且** `TOPICS="YES"`（有被標註主題） |
| ModLewis | 13,625 | 6,188 | 1,765 | `LEWISSPLIT` 為 TRAIN / TEST，`TOPICS` 是 YES 或 NO 都收；BYPASS 與 NOT-USED 不用 |
| ModHayes | 20,856 | 722 | 0 | 只看 `CGISPLIT`：TRAINING-SET / PUBLISHED-TESTSET |

**為什麼預設還是 ModApte？** README VIII.D 明確推薦：做 TOPICS（主題）相關研究用 ModApte，因為很多「應該有主題」的新聞其實沒被標註；ModLewis 只在需要測試「不屬於任何主題的新聞」時才用；ModHayes 有「主題分布不尋常、近似重複、時間過度集中」等問題。作業也指定 ModApte。我們的相關性正好是用 TOPICS 判斷，所以 ModApte 最合適。

下方「三種切分差在哪裡？」表列出每種切分的測試集裡 5 個主題各有幾篇、幾篇完全沒有標籤：

| 切分 | earn | acq | crude | grain | trade | 測試集無標籤 |
|---|---|---|---|---|---|---|
| ModApte | 1,087 | 719 | 189 | 149 | 117 | 280 |
| ModLewis | 1,088 | 719 | 189 | 149 | 117 | 3,166 |
| ModHayes | 22 | 79 | 54 | 45 | 65 | 157 |

- **ModLewis** 的相關新聞和 ModApte 幾乎一樣，但多出約 2,900 篇沒有標籤的新聞。它們一律被當成「不相關」，其中有些其實在講原油、穀物，只是沒人標註。
- **ModHayes** 的測試集只有 722 篇，集中在 1987/4/2–6/16；它的訓練集反而包含測試期之後的新聞。主題比例也跟其他兩個完全不同（earn 只剩 22 篇）。

右上角「依主題瀏覽全部新聞 →」會打開新聞列表（見最後的「新聞列表」一節）。

### 上方四個數字

| 數字 | 意思 |
|---|---|
| **訓練集 9,603** | 官方 README 定義的 ModApte 訓練集（`LEWISSPLIT="TRAIN"` 且 `TOPICS="YES"`）。**只拿來訓練 Word2Vec**。 |
| **測試集 3,299** | ModApte 測試集（`LEWISSPLIT="TEST"` 且 `TOPICS="YES"`）。**這 3,299 篇就是被搜尋的新聞**。README 要求不可刪掉任何一篇，所以全部保留。 |
| **測試集詞彙量 18,314** | 測試集前處理後，不重複的詞有幾個（共 259,353 個詞）。 |
| **平均長度 78.6** | 每篇新聞前處理後平均幾個詞。 |

為什麼要分開？如果 Word2Vec 也用測試集訓練，詞向量等於「看過」要被搜尋的新聞，結果會太樂觀。

### 5 個主題表

作業指定的 5 個主題，以及測試集中各有幾篇：

| topics | 主題 | 篇數 |
|---|---|---|
| earn | 企業獲利與財務結果 | 1,087 |
| acq | 企業收購與合併 | 719 |
| crude | 原油 | 189 |
| grain | 穀物 | 149 |
| trade | 國際貿易 | 117 |
| 其他 | 有其他 topics（例如 money-fx、ship），但不屬於這 5 個主題 | 777 |
| 無標籤 | 完全沒有 topics 標籤 | 280 |

「其他」和「無標籤」的新聞仍然留在檢索範圍裡，當作干擾項：系統要能避開它們。一篇新聞可能同時屬於多個主題。（以上是 ModApte；換切分後數字會跟著變。）

### 資料與前處理

- **來源**：UCI Machine Learning Repository 的 Reuters-21578，解析 22 個 SGML 檔（`reuters/corpus.py`）。
- **前處理**（`reuters/text.py`，TF-IDF 和 Word2Vec 共用同一套，詞彙才一致）：
  1. tokenize 切詞。
  2. 轉小寫。
  3. 去標點。
  4. 移除 stopword（the、of 這類功能詞）。
  5. 移除純數字（財報新聞有大量金額，對詞義沒幫助）。
  6. 不做 stemming（保留原詞形，鄰近詞比較好解讀）。
- **相關性**：新聞的 topics 含有查詢的目標主題就算「相關」（1），否則「不相關」（0）。

---

## ② Word2Vec 訓練：CBOW 與 Skip-Gram

### 兩張架構說明卡

- **CBOW（Continuous Bag of Words）**：用**上下文**猜**中間的詞**。例：`shipment of ___ to japan` → wheat。把上下文詞的向量平均起來做預測，一個視窗只預測一次，所以訓練快，對常見詞比較穩定。
- **Skip-Gram**：反過來，用**中間的詞**猜**每一個上下文詞**。例：wheat → shipment、of、to、japan。一個視窗要預測好幾次，訓練較慢，但少見的詞也能得到比較多次更新。

兩者訓練完，每個詞都會得到一個 100 維的向量。上下文相似的詞，向量方向也會相近。

### 訓練結果表

| 項目 | CBOW | Skip-Gram | 說明 |
|---|---|---|---|
| 訓練句數 | 51,188 | 51,188 | 9,603 篇訓練集切成的句子 |
| 訓練詞數 | 805,875 | 805,875 | 兩者用完全相同的資料（控制變因） |
| 詞彙量 | 14,679 | 14,679 | 出現至少 3 次的詞才學向量 |
| 訓練時間 | 21.7 秒 | 70.1 秒 | Skip-Gram 每個視窗要預測多次，所以較慢 |
| 最後一個 epoch 的 loss | 540,084 | 2,122,592 | 兩者 loss 的定義不同，**不能直接比大小** |

**共用參數**（兩種架構完全相同）：

| 參數 | 值 | 意思 |
|---|---|---|
| 維度 | 100 | 每個詞向量的長度 |
| 視窗 | 5 | 中心詞左右各看 5 個詞當上下文 |
| 最低詞頻 | 3 | 少於 3 次的詞不學向量（太少見學不好） |
| negative sampling | 5 | 每次預測時，隨機抽 5 個「錯誤答案」來對照，加快訓練 |
| epochs | 20 | 整份語料訓練 20 輪 |
| seed | 42 | 固定隨機種子，結果可重現 |

### 「自己用 numpy 實作的 CBOW / Skip-Gram」（點一下展開）

這是回應學習目標「理解並實作 CBOW 與 Skip-Gram 的基本訓練流程」。真正的實驗用 gensim，這裡另外用 numpy 從頭寫了一個看得懂每一步的版本（`reuters/toy_word2vec.py`）：

- **示範語料**：16 句模仿 Reuters 的短句，詞彙 40 個、訓練樣本 77 個。
- **一個訓練樣本的例子**：上下文 `opec, raised, oil, output` → 中心詞 `crude`。
- **每一步的流程**：
  1. 算出隱藏層向量。CBOW 是上下文向量的平均；Skip-Gram 是中心詞的向量。
  2. 用 softmax 算出每個詞是答案的機率。
  3. loss = −log（正確答案的機率）。
  4. 用梯度下降更新兩個權重矩陣。
- **表格怎麼看**：第 1 個 epoch 的 loss 是 3.69，訓練後 CBOW 降到 0.25、Skip-Gram 降到 1.9，代表模型確實在學。Skip-Gram 的 loss 比較高是正常的：同一個中心詞要同時猜好幾個不同的上下文詞，本來就不可能每個都猜中。
- **重點觀察**：profit 和 loss 在示範語料裡的上下文完全一樣（`company reported net ___`），所以兩者的 cosine 是 **1.00**。向量一模一樣，意思卻相反。
- 程式的正確性有單元測試保證：梯度和數值微分的結果一致。

### 訓練 loss 曲線

gensim 每個 epoch 的 loss 都往下降，代表訓練有在收斂。

---

## ③ 詞層級分析：Top-5 鄰近詞

最上面的說明是整段的重點：**向量相近不代表兩個詞是同義詞，只代表兩個詞常出現在相似的上下文**。

### 查一個詞的 Top-10 鄰近詞

輸入任何一個詞（例如 oil），左右並排顯示 CBOW 和 Skip-Gram 的 10 個最近的詞與 cosine 值。cosine 介於 −1 和 1 之間，越接近 1 越相似。

如果顯示「不在 Word2Vec 詞彙表中」，代表這個詞在訓練集出現少於 3 次，或是 stopword、數字。

### 「試試這些詞」按鈕

輸入框下方有 18 個可以直接點的詞，依「看得到什麼現象」分成五組（滑鼠移到詞上會提示要觀察什麼）。輸入框仍然可以自己打任何詞。

| 分組 | 詞 | 會看到什麼（ModApte 模型） |
|---|---|---|
| 反義詞也很近 | profit、rise、higher、surplus、buy | profit → loss（0.74）、rise → fall（0.83）、higher → lower（0.81）、surplus → deficit（0.83）、buy → sell（0.71）：意思相反，但出現在句子裡同一個位置 |
| 同一類的詞 | january、monday、yen、gold | 月份、星期、其他貨幣（marks、guilders）、其他貴金屬（silver、platinum）互相靠近 |
| 人物與組織 | opec、reagan、coffee | opec → lukman、nazer（石油部長）；reagan → administration、congress；coffee → ico（國際咖啡組織） |
| 1987 年的新聞事件 | japan、iran、strike | japan → microchips（美日晶片爭端）；iran → missiles、silkworm（兩伊戰爭的飛彈）；strike → seamen、miners（罷工） |
| 固定詞組與財報縮寫 | wall、palm、shr | wall → street（0.94，Wall Street）；palm → rbd、deodorised（棕櫚油規格）；shr → oper、vs、div（財報表格縮寫） |

### 13 個詞的 Top-5 鄰近詞表

涵蓋 5 個主題的詞，例如 oil、wheat、merger、profit、tariff、japan。每個鄰近詞後面的小數字是 cosine。

### 三個說明框（回答作業的三個問題）

1. **哪些鄰近詞有合理的語意關係？**
   wheat → corn、barley、grain；merger → acquisition、buyout；oil → crude、petroleum、gas；dividend → dividends、payout。
2. **哪些只是因為常出現在同類新聞？**
   - profit 最近的鄰居就是 **loss**：意思相反，但都出現在財報新聞的同一個位置。
   - opec → lukman、nazer：這兩位是當時奈及利亞和沙烏地阿拉伯的石油部長，因為常跟 OPEC 出現在同一則新聞。
   - japan → microchips：1987 年美日晶片貿易爭端。
3. **哪些不符合預期？為什麼？**
   - oil → kernel：來自 palm kernel oil（棕櫚仁油）。
   - Skip-Gram 常把很少見的專有名詞排進來，例如 profit → swissair、tariff → krapels、merger → siebe。原因是 Skip-Gram 讓少見的詞也得到很多次更新，只要它和某個詞出現在同一篇新聞幾次，向量就會被拉近。

### 互動散布圖「誰跟誰比較近？」

把 13 個種子詞（大圓點、粗體）和它們的 Top-5 鄰近詞（小圓點），從 100 維壓到 2 維畫出來。顏色代表種子詞屬於哪個主題。

**操作方式**：

| 操作 | 效果 |
|---|---|
| 滾輪、或右上的 ＋／− | 縮放（右上會顯示目前倍率） |
| 按住拖曳 | 平移 |
| 雙擊 | 放大 |
| 重設 | 回到原始大小 |
| 滑鼠移到點上 | 顯示這個詞、它是哪個種子詞的鄰近詞、cosine 是多少，並把同一組的詞標亮、其他變淡 |
| 點圖例（earn、acq…） | 隱藏或顯示某個主題 |
| CBOW／Skip-Gram | 切換架構 |
| t-SNE／PCA | 切換投影方法 |

**放大時字的大小不會變，只有點和點之間的距離被拉開**，所以原本疊在一起的詞會分開。一開始因為太擠而藏起來的標籤，放大後空間夠了就會出現。實測：一開始顯示 27 個標籤，放大 3 倍後顯示 59 個。

**兩種投影怎麼看**：
- **t-SNE**：保留「誰跟誰很近」的局部結構，5 個主題的詞各自成群；但**群與群之間的距離沒有意義**。
- **PCA**：保留整體變異最大的兩個方向，群與群之間的相對位置比較可信，但局部容易擠在一起。
- 兩種都只是 100 維的 2D 近似，真正的相近程度以表格裡的 cosine 為準。

---

## ④ 新聞搜尋：Top 10

### 搜尋表單

| 欄位 | 說明 |
|---|---|
| 查詢 | 輸入英文查詢，例如 `petroleum supply from the gulf`；可以直接打字、修改、刪除 |
| 方法 | 7 種，見下表 |
| 目標主題 | 選一個主題後，屬於這個主題的結果會標綠色（相關），方便一眼看出排得好不好；選「不標示」就不標 |
| 評估用的 20 個查詢 | 表單下方依主題列出全部 20 個查詢，點一下就用它搜尋並自動選好目標主題（方法維持原本選的） |

### 評估用的 20 個查詢（為什麼是這些？）

Reuters-21578 是**文字分類**資料集，只附主題標籤，**沒有官方查詢**，所以查詢是自己設計的，每個主題 4 個：

- **關鍵字（keyword）**：用該主題新聞**最常見、又很少出現在其他主題**的詞來寫，是 TF-IDF 的主場。每個主題旁邊的「常見詞」就是從訓練集統計出來的：含這個詞的新聞比例（主題內）減掉（主題外），越大越有代表性。
- **換說法（paraphrase）**：刻意避開最典型的那幾個詞，改用同義或相關的說法，用來測試詞向量能不能補上「字不一樣就找不到」的問題。

ModApte 訓練集統計出來的「常見詞」前 8 名：

| 主題 | 常見詞 | 對應的關鍵字查詢 |
|---|---|---|
| earn | vs、cts、net、shr、qtr、revs、note、loss | E1 quarterly **net** profit and earnings per share |
| acq | company、inc、shares、said、corp、acquisition、group、stake | A1 merger and **acquisition** agreement、A2 takeover bid tender offer for **shares** |
| crude | oil、crude、barrels、petroleum、prices、energy、day、production | C1 **crude oil prices** and OPEC output、C2 **oil production** in **barrels** per **day** |
| grain | wheat、tonnes、agriculture、grain、u.s.、corn、usda、export | G1 **wheat** and **corn** exports、G2 **grain** shipments to the soviet union |
| trade | trade、u.s.、he、japan、exports、deficit、surplus、countries | T1 **trade deficit** with **japan** |

所以關鍵字查詢基本上就是「這個主題最常見的詞組合」，換說法查詢則故意不常見。要注意：換說法的查詢不一定完全避開這份清單，例如 C3 用的 petroleum、C4 用的 energy 也在原油的前 8 名；但 petroleum 也常出現在公司名稱（British Petroleum），所以 TF-IDF 在 C3 還是被誤導。之前畫面上只放了 5 個當範例，現在 20 個全部列出來，可以直接選。

**7 種方法**：

| 方法 | 怎麼算 | 角色 |
|---|---|---|
| **A · TF-IDF** | 每篇新聞是一個「詞 × 權重」的向量，權重 = tf × idf，idf = log₁₀(N / df)；查詢也一樣；分數 = 兩個向量的 cosine | 作業必做 |
| **B · TF-IDF weighted CBOW** | 每篇新聞 = 文中每個詞的 CBOW 向量，用 tf × idf 當權重做**加權平均**；查詢也一樣；分數 = cosine | 作業必做（核心比較） |
| **B · TF-IDF weighted Skip-Gram** | 同上，改用 Skip-Gram 向量 | 作業必做（核心比較） |
| C · Mean CBOW／Skip-Gram | 不加權，直接平均所有詞向量 | 額外比較：看「TF-IDF 加權」本身有沒有幫助 |
| D · Hybrid | 0.5 × A 的分數 + 0.5 × B 的分數 | 額外比較：常見的折衷做法 |

### 查詢詞診斷表

列出查詢經過前處理後剩下的每個詞：
- **在測試集中**：TF-IDF 只能用測試集裡出現過的詞。打 ✗ 的詞，TF-IDF 會直接忽略。
- **DF／IDF**：這個詞出現在幾篇新聞、稀有程度多高。
- **有詞向量**：詞向量方法只能用 Word2Vec 詞彙表裡的詞。

這張表可以解釋為什麼某個方法找不到東西：例如查詢詞根本不在測試集裡時，TF-IDF 就無從比對。

### 搜尋結果 Top 10

每一筆顯示：
- 排名
- **標題（可以點進新聞內頁）**
- NEWID 和日期
- 主題標籤
- 摘要片段（黃色是查詢詞）
- 分數（cosine）

選了目標主題時，**左邊有綠色直條的就是相關的新聞**，不相關的會稍微變淡。

### 方法 A vs 方法 B 並排

同一個查詢，A、B-CBOW、B-Skip-Gram 三欄並排 Top 10。標題列寫「相關 x/10」，代表前 10 名有幾篇屬於目標主題；相關的列會有淡綠底。每個標題一樣可以點進新聞內頁。

**範例**：查 `petroleum supply from the gulf`、目標主題 crude：

| 方法 | 前 10 名相關篇數 | 原因 |
|---|---|---|
| A · TF-IDF | 4/10 | 抓到很多 Gulf 戰爭的航運新聞（主題是 ship），以及公司名稱「British Petroleum」的股票新聞；字面相同，主題不對 |
| B · CBOW | 9/10 | 詞向量知道 petroleum 和 crude、oil、barrels 很接近 |
| B · Skip-Gram | 8/10 | 同上 |

### 新聞內頁（點標題進入）

| 區塊 | 內容 |
|---|---|
| 標題列 | NEWID、日期、**主題**（TOPICS 標籤 + 中文名稱，例如「crude 原油」；沒有標籤的寫「未分類」）、**所屬切分**（這篇在 ModApte / ModLewis / ModHayes 各是訓練集、測試集還是未使用；目前切分用藍框標出）、前處理後的詞數 |
| 查詢區塊（從搜尋點進來才有） | 這篇對目標主題是相關還是不相關，以及**這篇在三種方法的完整排名中排第幾名**。例：上面的原油新聞，TF-IDF 排第 48 名、CBOW 第 2 名、Skip-Gram 第 1 名。如果這篇不在目前切分的測試集（例如從新聞列表點進一篇訓練集新聞），會說明它不會被檢索、沒有排名 |
| 全文 | 查詢詞標成黃色 |
| TF-IDF 權重最高的 10 個詞 | 方法 A「看到」的就是這些詞，tf × idf 越高越能代表這篇；點詞可以跳回主頁看它的鄰近詞 |
| ← 回到搜尋結果 | 回到原本的查詢和設定（從新聞列表進來的話是「← 回到新聞列表」） |

### 新聞列表（依主題瀏覽全部新聞）

網址 `/reuters/news/`，從 ① 右上角的「依主題瀏覽全部新聞 →」或「文章」頁上方的 Reuters-21578 入口進來。

- **這些新聞叫什麼類別？** 資料集名稱就是 **Reuters-21578**（和 PubMed 文章庫分開存放，所以 HW1–Project 3 的搜尋、Zipf、權重實驗都不受影響）。每篇新聞的「主題」就是它原本的 TOPICS 標籤（earn、acq、crude…，共 120 種在用，一篇可以有好幾個），常見的標籤都附中文名稱。
- **篩選**：切分（ModApte / ModLewis / ModHayes）× 範圍（全部 21,578 篇 / 訓練集 / 測試集 / 未使用）、標題關鍵字。
- **主題**：「全部」、作業的 5 個主題、「無 TOPICS 標籤」各是一個按鈕（附篇數）；其他 115 種主題（外匯、利率、航運、咖啡…）在「其他主題」下拉選單，選了就直接套用。
- 每頁 25 篇，標題點進去就是新聞內頁。

---

## ⑤ 文件層級評估

### 查詢集

20 個查詢，5 個主題各 4 個（`reuters/queries.py`）：
- **keyword**（每主題 2 個）：用該主題最典型的關鍵字，例如 `crude oil prices and OPEC output`。這是 TF-IDF 的主場。
- **paraphrase**（每主題 2 個）：刻意換說法、避開典型關鍵字，例如 `energy ministers discuss fuel pricing`。這是測試詞向量的地方。

### 指標說明（點一下展開）

| 指標 | 意思 |
|---|---|
| P@10 | 前 10 筆有幾成相關 |
| MRR | 第一篇相關新聞排第幾名取倒數，再對 20 個查詢平均 |
| nDCG@10 | 相關新聞排越前面分數越高，除以理想排序的分數；1 = 完美 |
| Recall@10 | 所有相關新聞中有幾成出現在前 10 名；每個主題有 117–1,087 篇相關，上限很低，僅供參考 |

### 平均結果表（20 個查詢）

| 方法 | P@10 | MRR | nDCG@10 |
|---|---|---|---|
| **A · TF-IDF** | 0.730 | 0.833 | 0.728 |
| **B · TF-IDF weighted CBOW** | 0.850 | 0.910 | 0.848 |
| **B · TF-IDF weighted Skip-Gram** | 0.895 | **0.967** | **0.908** |
| C · Mean CBOW | 0.865 | 0.885 | 0.858 |
| C · Mean Skip-Gram | **0.905** | 0.925 | 0.900 |
| D · Hybrid A + CBOW | 0.835 | 0.887 | 0.836 |
| D · Hybrid A + Skip-Gram | 0.810 | 0.917 | 0.823 |

粗體列是作業要求的核心比較（A vs B），粗體數字是各指標最高。重點：
- 兩種 B 都比 A 好，Skip-Gram 最好。
- C（不加權）和 B 差不多，代表在這組資料上「TF-IDF 加權」的幫助不大。
- D（混合）反而比單用 B 差。可能原因是兩邊分數的範圍不同：
  - TF-IDF 前 10 名的 cosine 是 0.06–0.49（平均 0.23），沒有共同詞的新聞是 0，差異很大。
  - Skip-Gram 前 10 名是 0.67–0.90（平均 0.77），連一般新聞都有 0.44 左右，擠在窄範圍內。
  - 直接相加時，排名主要由差異大的 TF-IDF 決定，混合法就變得比較像 TF-IDF。

### keyword vs paraphrase（nDCG@10）

| 方法 | keyword | paraphrase |
|---|---|---|
| A · TF-IDF | 0.936 | **0.520** |
| B · CBOW | 0.910 | 0.785 |
| B · Skip-Gram | 0.971 | **0.845** |

**差距幾乎都來自換說法的查詢**。用典型關鍵字時，TF-IDF 和詞向量差不多；換說法時，TF-IDF 因為字面對不上，掉到 0.52。

### Word2Vec 訓練語料加大

README 說「未使用的 8,676 篇可以用來收集統計資訊」，所以額外試了：Word2Vec 改用「訓練集 + 未使用」共 18,279 篇訓練，**只用來訓練詞向量，不參與檢索**。

| 方法 | 只用訓練集 | 加上未使用 |
|---|---|---|
| B · CBOW | 0.848 | 0.867 |
| B · Skip-Gram | 0.908 | 0.921 |

語料變大，兩者都小幅進步。

### 逐查詢 nDCG@10 表

每一列是一個查詢，比較 A、B-CBOW、B-Skip-Gram：
- 顏色越深代表在這題越好，框起來的是這題最好的方法。
- 最右欄是「Skip-Gram − TF-IDF」：綠色代表詞向量贏、紅色代表 TF-IDF 贏。
- 點查詢文字會用該查詢搜尋，可以直接比較結果。

值得注意的幾題：

| 查詢 | TF-IDF | Skip-Gram | 說明 |
|---|---|---|---|
| T4 dispute over foreign goods entering markets | 0.18 | 1.00 | 換說法，詞向量大勝 |
| T3 protectionism between nations | 0.18 | 0.80 | 同上 |
| A3 firm buys a rival business | 0.39 | 1.00 | 同上 |
| **E4 firm reports bigger losses than last year** | **0.82** | **0.26** | **例外：TF-IDF 反而好很多**，是後續要深入分析的題目 |

### 兩張圖

- **熱圖**：把平均結果表畫成顏色，每一欄一個指標，框起來的是各指標第一名。
- **keyword vs paraphrase 圖**：每個方法兩個點，藍色是關鍵字查詢、橘色是換說法查詢。兩點距離越長，代表這個方法越依賴字面比對。TF-IDF 的距離最長。

### 三種切分的檢索結果（第 5 段最後一張表與圖）

同樣 20 個查詢、同樣的方法，只換切分；每個切分的 Word2Vec 都用它自己的訓練集重新訓練。表格裡淺藍底的欄位是目前選的切分。

| 方法 | nDCG@10 ModApte | ModLewis | ModHayes | 換說法 nDCG@10 ModApte | ModLewis | ModHayes |
|---|---|---|---|---|---|---|
| A · TF-IDF | 0.728 | 0.531 | 0.580 | 0.520 | 0.278 | 0.386 |
| B · CBOW | 0.848 | 0.658 | 0.662 | 0.785 | 0.570 | 0.527 |
| B · Skip-Gram | **0.908** | 0.732 | 0.675 | **0.845** | 0.648 | 0.537 |

（P@10：A 0.730 / 0.545 / 0.555，B-Skip-Gram 0.895 / 0.745 / 0.670；MRR：A 0.833 / 0.614 / 0.772，B-Skip-Gram 0.967 / 0.800 / 0.786）

**結論一：不論哪個切分，B 都比 A 好**，而且差距都主要來自換說法的查詢。所以「詞向量能補上字面不同」這個結論不是 ModApte 才成立。

**結論二：ModLewis 的分數全部掉了，但不是方法變差，而是「答案卷」變了。** 它的 Word2Vec 用了更多訓練資料（13,625 篇），可是測試集多了約 2,900 篇沒有 TOPICS 標籤的新聞，這些一律算不相關。20 個查詢的前 10 名共 200 個位置裡，被這種沒標籤新聞佔走的：

| 切分 | TF-IDF：相關 / 無標籤 / 其他主題 | B-Skip-Gram：相關 / 無標籤 / 其他主題 |
|---|---|---|
| ModApte | 146 / 3 / 51 | 179 / 4 / 17 |
| ModLewis | 109 / **58** / 33 | 149 / **42** / 9 |
| ModHayes | 111 / 35 / 54 | 134 / 26 / 40 |

例如 ModLewis 裡查 E2 `dividend declared to shareholders`，TF-IDF 前 10 名有 7 篇是 TOPICS = NO 的新聞，標題卻是「BRISTOL-MYERS CO DECLARES DIVIDEND」「CHANCELLOR DECLARES SPECIAL DIVIDEND」，明明就是財報 / 股利新聞，只是當年沒被標註。這正是 README 說的「很多應該有主題的新聞沒被標」，也是 ModApte 把它們排除的原因。

**結論三：ModHayes 的 A、B 差距最小（nDCG@10 只差 0.10，ModApte 差 0.18）。** 測試集只有 722 篇，主題比例很特別：earn 只有 22 篇，earn 查詢在 ModHayes 上 TF-IDF 只有 0.27、Skip-Gram 只有 0.15。相關新聞太少時，前 10 名本來就很難填滿，加上 README 提到的近似重複與時間集中，數字比較不穩定，不適合當主要結論。

**為什麼預設仍是 ModApte**：它的相關性判斷最可靠（只用有標註主題的新聞），README 推薦、作業指定，結果也最能反映方法本身的差異；ModLewis 和 ModHayes 適合當「換一種切法，結論還成立嗎」的補充。

---

## 結果解讀時要注意的限制

1. **相關性是用「主題」判斷的**：平均詞向量本來就擅長抓「這篇在講哪個領域」，所以這種評估方式可能偏向詞向量。如果改成更細的人工判斷（例如「真的在講美日貿易逆差」），結果可能不同。
2. **加權平均會失去詞序與否定**：`profit rose` 和 `profit fell` 的平均向量幾乎一樣。這可能是 E4 這類查詢詞向量表現差的原因之一（待分析）。
3. **小型語料**：Word2Vec 只用 80 萬個詞訓練，屬於小型語料，結果不能直接推論到大型語言模型。
4. **20 個查詢**是小樣本，沒有做統計顯著性檢定。

## 重跑實驗

```powershell
docker compose exec web python manage.py run_reuters_experiments                   # 三種切分全跑：docs/project4_results/<切分>/ 的 CSV、PNG、projection.json，加上最上層的 split_comparison
docker compose exec web python manage.py run_reuters_experiments --split modapte   # 只重跑一個切分（比較表會沿用其他切分之前的結果）
docker compose exec web python manage.py test reuters                              # 單元測試
```
