# Project 5 報告：可以補充與修正的地方

對照 Homework-5 的要求，報告（BIR_hw5）目前還沒寫到的部分，以及幾句和實驗數字不完全一致的地方。數字都來自 `docs/project5_results/`。

## 一、建議修正（數字或說法不精確）

| 原句 | 問題 | 建議改成 |
|---|---|---|
| 排序是 BSBI 的主要成本（0.59 秒） | 前處理 2.33 秒才是最久的 | 除了前處理之外，排序是 BSBI 最花時間的步驟（0.59 秒，約佔建索引時間的 1/3） |
| 只排序這個 block 的詞彙（約 19,867 個 terms，0.03 秒） | 0.03 秒是 7 個 blocks 的排序合計 | 只排序這個 block 的詞彙（block 1 約 19,867 個 terms；7 個 blocks 合計只要 0.03 秒） |
| medical research 等查詢在 1987 年財經新聞中幾乎不會出現 | financial market 的 AND 有 453 篇，並不少 | 作業範例的 information retrieval、computer science、artificial intelligence、medical research 的 AND 分別只有 2、3、11、16 篇（financial market 有 453 篇），因此改用符合 Reuters 內容的查詢 |
| Intermediate files：block 越多，寫檔與合併時要讀的檔越多 | 兩者總大小差不多（6.3 vs 6.1 MB），差在檔案數 | 總大小差不多，但 BSBI 有 27 個檔、SPIMI 只有 7 個，合併時要同時開的檔與讀取緩衝區較多 |

## 二、作業有要求、報告還沒寫到的

1. **前處理方式**（Part 1 第 2 點）
   > 文件為 Reuters-21578 全部 21,578 篇（標題 + 內文，docID = NEWID）。前處理：HW1 的 tokenizer、全部轉小寫、移除 72 個 stopword 與沒有英文字母的 token，不做 stemming。共 1,730,348 個 tokens、51,470 個 terms、1,169,501 筆 postings。

2. **作業第 4 題：記憶體有限時怎麼處理**（報告沒有改變記憶體上限的實驗）
   > 兩者都是記憶體滿了就把局部索引寫到磁碟、清空，最後再合併。把上限從 64 MB 降到 1 MB，BSBI 的 blocks 從 2 個變成 106 個、SPIMI 從 1 個變成 50 個。但上限不是越小越好：1 MB 時 BSBI 要花 4.93 秒（8 MB 時 4.05 秒），峰值反而升到 13.3 MB，因為合併時要同時開 106 個檔，每個檔都有讀取緩衝區。BSBI 的峰值每次都超過上限，因為 termID 對照表（4.2 MB）不能寫出去；SPIMI 沒有這張表，2 MB 上限時峰值只有 2.5 MB。

3. **作業第 6 題：文件數增加到 100,000、1,000,000**（報告沒有規模實驗）
   > 4 MB 上限下，10,000 → 100,000 → 1,000,000 篇：BSBI 2.27 → 21.3 → 216 秒、SPIMI 1.72 → 16.9 → 171 秒，時間大致和文件數成正比。blocks 也跟著變多（BSBI 13 → 123 → 1,223 個；SPIMI 3 → 30 → 290 個），1,000,000 篇的中間檔約 291 MB / 280 MB，合併要同時開上千個檔。BSBI 的 termID 對照表會隨詞彙量變大，依 Heaps' law 推估 1,000,000 篇約 38 萬個詞、對照表約 31 MB。註：超過 21,578 篇的部分是重複讀語料，不會有新詞。

4. **Baseline（Part 1 說要「作為後續 BSBI 與 SPIMI 的比較基準」）**
   > 不限記憶體、整份索引放在記憶體建好再一次寫出：3.18 秒，峰值 18.5 MB，比 BSBI（4.13 秒）和 SPIMI（3.32 秒）都快，但記憶體會跟著文件數一直變大（100,000 篇 56.3 MB，SPIMI 維持 4.6 MB）。三個索引逐一比對，51,470 個 terms 的 postings 完全相同。

5. **怎麼量的**（作業要求根據實驗數據）
   > 時間用 time.perf_counter，每種方法跑 5 次取中位數；記憶體用 tracemalloc 量峰值，另外跑一次（tracemalloc 會拖慢速度）；記憶體上限用 Python 物件的實際大小（sys.getsizeof）估計。

6. **檢索測試的「single-term」**：表格的 term 1、term 2 兩欄就是 single-term query 的命中數，可以在表頭註明「（single-term）」。

7. **第 1、2、3、5 題**：報告的 Part 2、Part 3 與比較表其實已經回答了，只是沒有標題號。如果老師是照題號看，可以在段落前加「（第 1 題）」之類的標示，或另外列一個 6 題的簡答。網頁第 5 段的 toggle list 有完整的 6 題回答可以參考。
