# BioMed IR — Docker Literature Workspace

Based on [c1air3tsai/BIR main](https://github.com/c1air3tsai/BIR/tree/main), commit `fdd2df638b1993b7eeed314351b120f4ff56858f`. The original HW1 abstract processing and blue palette are retained.

## Start on Windows

Install and start Docker Desktop with Linux containers. Extract this folder separately from your old installation. Open PowerShell in the folder containing `docker-compose.yml`.

Export this Windows computer's **already trusted public root certificates** once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\prepare_certs.ps1
```

The script only reads valid certificates in CurrentUser/Root and LocalMachine/Root and writes public PEM files into `certs/`. It does not change Windows trust settings or export private keys. These files let Docker verify HTTPS when a school, company or security product uses a locally trusted proxy certificate. Repeat this step if the network's trusted certificates change.

Build and start from the terminal:

```powershell
docker compose up --build
```

Open **http://127.0.0.1:8001/** yourself. There is no automatic browser opening. Keep the terminal running; Ctrl+C stops the app. Future starts use `docker compose up`. `docker compose up -d` is optional for background operation; `docker compose down` stops that mode and preserves the data volume.

The host defaults to port **8001**, avoiding the Windows permission failure previously reported on port 8000. If your computer also reserves 8001, copy `.env.example` to `.env`, set `BIOMEDIR_PORT=8888`, and use http://127.0.0.1:8888/.

Docker installs Python dependencies, applies database migrations and initializes the original 14 XML articles when the collection is empty. Imports run in a separate Python worker in the same container. A restart keeps collected articles and marks unfinished jobs Interrupted; request the same topic again to add more articles.

## HTTPS certificate correction

The downloader uses `truststore` for native certificate verification and loads trusted `.crt` / `.pem` CA files from `certs/`. The Docker build installs those CA certificates before downloading Python packages. Runtime requests retain certificate-chain and hostname verification. Certificate errors identify the actual cause and the recovery step instead of reporting a generic internet failure.

Check search and one abstract download after startup:

```powershell
docker compose exec web python manage.py ncbi_check
```

If it still reports `CERTIFICATE_VERIFY_FAILED`, Windows may not trust that proxy CA either. Obtain the **trusted CA certificate** from your network administrator, save a PEM `.crt` or `.pem` file in `certs/`, and rebuild with `docker compose up --build --force-recreate`. Do not save a website leaf certificate as a trusted root. Do not disable SSL verification. Optional `HTTP_PROXY` / `HTTPS_PROXY` values in `.env` can be used when your network requires an explicit proxy.

All instructions for article import and topic selection are also in **使用說明.md**.

## Update an existing installation after `database is locked`

Keep your current project folder and its Compose project name. Stop it with Ctrl+C or `docker compose down` (without `-v`). Copy the updated archive contents over the **same existing folder**, keeping your `certs/*.crt`, `certs/*.pem`, `.env`, and data. Then run:

```powershell
docker compose up --build --force-recreate
```

The existing named volume is reused; migrations and initialization retain its collection. Do not delete the volume, database, or old folder. If an import stopped at **199 / 500**, the 199 topic articles remain. After updating, submit **GLP-1, 301** to request the additional articles needed for a 500-article topic. Submitting 500 again requests 500 additional articles.

SQLite now uses WAL journaling and IMMEDIATE transactions. Status polling does not issue an update unless a stale job actually exists. Short write contention retries outside the complete transaction; a rolled-back article, topic link, postings and counters are retried together. This prevents duplicated counts or orphan XML when a record is retried. Network calls remain outside write transactions. Retry attempts are bounded, and other database errors are reported.

For a running WAL database, do not back up only the main `.sqlite3` file: recent changes may still be in the WAL file. Stop the application and workers before a volume-level backup, or use SQLite's backup API.

## Import articles

Upload has three entry points:

1. **Import by topic**: enter a PubMed query such as GLP-1 and request 1–1,000 articles. The query becomes the topic label. English records with a nonempty abstract are searched in relevance order and fetched through NCBI E-utilities, in batches of at most 50. The existing HW1 XML parser, sentence segmentation, statistics and abstract index are reused.
2. **Upload XML**: select up to 20 individual PubMed / PMC XML files, optionally assigning a topic label.
3. **Specific article IDs**: enter up to 20 PMIDs or PMCIDs. Plain digits mean PMID; PMCIDs use the PMC prefix. Fetch XML, then use Add article / Add all to collection.

The background job shows progress and new, reused, duplicate and skipped counts. Existing documents found in another topic are linked to the new topic; their abstract and database row are reused. Records already in the requested topic are skipped. Repeating a topic import requests **additional** articles. Stop, failed and interrupted jobs retain completed work. Stop takes effect after the current NCBI request. Only one topic import runs at a time; it scans at most the first 10,000 search matches to find eligible articles. Smaller result sets may produce a partial import. There is no guarantee that every NCBI search match has usable XML or an abstract.

Duplicate identity uses PMCID, PMID or DOI. Filename and title/year are fallbacks for records without stable IDs. Topics are normalized for spacing and capitalization. Automatic topic imports require PMIDs; older manually uploaded PMC-only articles remain usable.

## Common filters and pagination

**Search, Articles, and Zipf Analysis** share a topic row fixed below the main Search / Articles / Zipf / Upload navigation. Both rows stay visible when scrolling. Topics scroll horizontally when there are many labels, keeping the toolbar compact:

- No topic selected: all articles, including uncategorized articles.
- One selected: that topic.
- Multiple selected: the **union** of those topics. Overlapping articles are counted once.

Select topic chips; the page updates automatically after a short pause; **Apply** remains available. All articles clears the selection; the page automatically shows everything. Search and the page's filter controls also use the visible topic selection. Navigation retains the applied topics. Changing topics resets pagination while retaining the current search, publication year, sorting and Zipf condition. Articles and Search Results display **20 records per page**. Term tables also display 20 rows per page, with the full Top 50 accessible over three pages.

Articles list metadata and Abstract Statistics (Sentences → Words → Characters). Search Results add abstract snippets and keyword matches. Article Detail preserves sentence segmentation and Previous / Next keyword navigation. Existing HW1 biomedical tokenization rules are retained: GLP-1, COVID-19, decimals, percentages, Unicode letters, abbreviations and apostrophes. The searchable corpus is **Abstract + Keywords**, excluding bibliographic titles and hidden full text. PMC body text remains stored for future work.

## Zipf dashboard

One dashboard uses the same selected corpus for four cumulative conditions:

| Condition | Processing |
| --- | --- |
| A | HW1 word tokenization + Unicode casefold; standalone punctuation excluded |
| B | Explicit punctuation exclusion; equals A because tokenization already excludes it |
| C | B minus the shared HW1 stopword list |
| D | C + NLTK Porter stemming |

A–D tabs show corpus statistics, rank–frequency and log–log plots, regression, regional fits, terms and stemming examples. Both plots appear side by side, and checkboxes control which A–D lines are visible in both charts. Compare All initially overlays all four distributions and reports their tokens, vocabulary, exponent, R², RMSE and most frequent terms. The line selection is independent of the term table: in Compare All the table remains explicitly Condition D; select an A–D tab to inspect another vocabulary.

Calculations:

- **CF**: occurrences across all abstracts.
- **DF**: number of distinct abstracts containing the term.
- **IDF**: `ln(N / DF)`, without smoothing; N excludes blank abstracts.
- **Rank**: descending CF, alphabetical tie-breaking.
- **Fit**: unweighted ordinary least squares, `ln(CF) = intercept + slope × ln(rank)` over every vocabulary entry.
- **Zipf exponent k**: negative slope.
- **RMSE**: error in natural-log frequency space.
- **Regions**: head first 5%, middle 5–50%, tail last 50% of vocabulary ranks; recalculated per condition. Local R² is undefined for a constant-frequency segment. Regional error against the overall fit and local fits are reported separately.

Small / empty corpora remain usable; unavailable regression values show a dash. A high R² is not a proof of Zipf's Law. Charts sample log-spaced points for responsiveness; regression and exports always use all vocabulary rows. The interface shows rank–frequency and log–log together and does not include a residual chart. Each chart can be downloaded as SVG; all application CSS / JavaScript is local, without CDN dependencies.

**Export results** downloads a ZIP containing summary.csv, A–D full term tables, regional metrics, corpus.csv (IDs, metadata, topics and extracted abstracts), and methodology.json. Spreadsheet formula-like text receives a leading apostrophe for safe CSV opening; methodology.json explains how to remove it when programmatically reading those text cells.

## Persistent data and updates

The Docker named volume `biomedir_data` stores:

- `/app/data/db.sqlite3`: documents, topics, postings and import jobs.
- `/app/data/corpus/`: original and collected XML.
- `/app/data/fetched/`: staged article files.
- `/app/data/logs/`: worker logs.

The archive does not include a prepopulated database. A fresh volume is seeded from the original 14 GitHub XML articles. Updating or rebuilding the same Compose project retains the existing collection and topic memberships. Keep the extracted folder name/location stable so Compose uses the same project volume. Do not use `docker compose down -v` for normal shutdown: `-v` deletes the data volume.

This new project does not automatically import a separate old Windows installation. Keep that old folder and its database as a backup. Back up **the SQLite database and corpus together** while imports are stopped. `initialize_local` only seeds an empty collection; `build_index` preserves document IDs and topic links when rebuilding postings.

Optional NCBI contact email and API key can be set in `.env`. These are not required. Requests are rate-limited and retry temporary HTTP/network errors. All CSS, JavaScript and SVG charts are local. Only dependency installation and article fetching need internet access.

## Development checks

```powershell
docker compose exec web python manage.py check
docker compose exec web python manage.py makemigrations --check --dry-run
docker compose exec web python manage.py test search
```

`VALIDATION.json` records checks performed on this delivery. The verification environment has no Docker daemon, so it can validate the configuration, entrypoint syntax and application code but cannot claim an actual Docker image build. TLS trust is additionally checked against a local HTTPS server with a private test CA: an untrusted certificate fails, explicitly trusted certificates work, and the wrong hostname still fails.

This package covers the requested topic collection, shared filters, pagination and Zipf analysis. It does not include separate Word2Vec, edit-distance or assignment-report tasks.

References: [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25499/), [PubMed search help](https://pubmed.ncbi.nlm.nih.gov/help/), [truststore documentation](https://truststore.readthedocs.io/en/latest/), [Django SQLite transactions](https://docs.djangoproject.com/en/5.2/ref/databases/#database-is-locked-errors), and [SQLite WAL](https://sqlite.org/wal.html).

## Complete algorithm rules

See **ALGORITHM_RULES.md** for extraction, word/sentence/character rules, the complete stopword list, Porter, BM25, topic scopes, Zipf regression, deduplication, retry and storage rules. **STOPWORDS.txt** is the exact fixed set. This revision leaves `text_processing.py` byte-identical to the previous delivery. Topic selections now apply automatically; the selected Zipf A–D lines, regression toggle and term sorting are retained when the corpus changes.

If a large import stops, run `docker compose exec web python manage.py import_diagnostics` and report the revision and actual error. This release adds visible temporary-network cooldown retries and skips malformed single XML records; it does not claim that every possible failure has the same cause.


### Complete Project 2 workspace

Open **Analysis** to switch between Zipf distribution, **Compare domains**, **Word2Vec**, **Text matching**, and **Research report**. `REQUIREMENTS_CHECKLIST.md` maps every supplied assignment requirement to its screen/export.

For the Optional Challenge, import 500 medical abstracts with **PubMed** and 500 CS abstracts with **arXiv** (e.g. `cat:cs.IR` or `cat:cs.LG`). Assign them to the two independent topic groups in Compare domains. The default balances counts and removes shared documents. arXiv provides its own identifiers, not PMID; the report explicitly records that distinction. No PDF/full body is downloaded.

Word2Vec trains Gensim Skip-gram or CBOW on your selected abstract sentences; models and metadata persist under `data/embeddings/`. Choose preprocessing and model settings, press Train, enter a word, inspect cosine neighbors and a local PCA plot, then export vectors. Changing settings or topics selects the appropriate model rather than showing stale vectors. A changed corpus requires training again. Small corpora may have unreliable neighbors.

Research report includes RQ1–RQ5, regional-fit interpretation, CF/DF/IDF evidence, true per-document TF-IDF, a 300–500 word IR discussion, and a printable one-page Executive Summary. It is an evidence-linked draft, not a fabricated 1,000-document result. Review its wording and add your own plot observations and course references before submission.

Text matching compares Unicode code points with Levenshtein distance, optional NFC/NFKC and casefold, and closest spelling suggestions from the chosen corpus. It does not change the HW1 statistics or silently alter the existing BM25 index.


### 本版標點規則（依使用者最新要求）

獨立的 . , ; 等標點不作Zipf terms，A–D全部排除。A用原HW1 word-token斷詞與casefold，B再次明確排除獨立標點；因A已排除，A/B相同是預期結果。詞內GLP-1、COVID-19、apostrophe、decimal與percent仍沿用HW1完整token，不刪掉生醫詞連接。不改HW1字數句數字元算法。


CS CSV備用：arXiv API可能回429限流。可上傳本機UTF-8 CSV（headers id,title,abstract；id保留真實arXiv ID），最多1,000rows/20MB。版本尾碼去重、單列invalid略過並提示、不捏造PMID、不抓正文，仍沿用HW1摘要索引及統計。API429則有界重試後顯示真正原因，已完成文章保留。


## Search spelling suggestions (2026-10-04)

Search for `camcer`: if `cancer` occurs in your selected abstracts, **Did you mean?** appears below the search box with up to three clickable alternatives. Suggestions can also appear for partial matches such as `camcer treatment`. Your original query and results remain visible until you click a suggestion. The corrected search retains selected topics and publication year and resets pagination. Topic changes refresh both results and suggestions. Known corpus words, known stem variants, short abbreviations, stopwords and biomedical IDs are protected from replacement. Suggestions are spelling candidates, not semantic Word2Vec neighbors. Full distance, ranking, normalization and cache rules are in ALGORITHM_RULES.md.

Hyphenated term variants are included automatically at query time. For example, `glp1` also retrieves documents indexed with `GLP-1`, and `GLP-1` also retrieves `GLP1`; the page shows which extra variants were included. Equivalent variants contribute as one BM25 query concept rather than being double-counted. This does not rebuild or alter the stored HW1 tokens.

## Bilingual interface and Word2Vec PCA (2026-10-05)

Use the **EN / 繁中** control beside the BioMed IR brand to switch the interface between English and Traditional Chinese. The browser remembers the selection. Interface controls and guidance are localized while article titles, authors, journals and abstracts remain in their source language.

After a Word2Vec model is trained, its page automatically renders a PCA overview of the 40 most frequent model terms. Point size represents corpus frequency and hover text shows the term and count. The existing query-plus-neighbors PCA remains available after entering one model word. PCA distances are a two-dimensional approximation; neighbor ranking continues to use cosine similarity in the original vector space. Exports include `vocabulary_pca.json`.

Multi-series Zipf and domain-comparison charts use distinct blue, vermilion, green, and purple hues from a color-blind-friendly palette. Lines, regression fits, hover markers, legends, and condition labels share the same A–D color mapping.

Rank–frequency and log–log charts are displayed together in a responsive two-column layout. Use the A–D checkboxes to show any combination of preprocessing lines on both charts; domain comparison provides the same control for Domain A/B. Residual mode has been removed. The selection is retained while changing topics, conditions, term pages, or domain settings.

For an existing installation, overwrite source files in the same Compose project folder, preserve `.env`, trusted certificates and the data volume, then run `docker compose up --build --force-recreate`. Open http://127.0.0.1:8001/.
