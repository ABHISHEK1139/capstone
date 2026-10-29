# Dataset Inventory — capstone-mirror / data_new

All statistics below were **computed by actually parsing each downloaded file**
(transaction count, unique-item count, average transaction length, density =
avg-length / unique-items × 100). No statistic was copied from the web.
Density is a property of the file as stored (see format notes for conversions).

Total: **25 datasets** (all under `data_new/`), spanning sparse retail,
dense, click-stream, synthetic, very small categorical, census, and
document-word sets.

**Added in extension (2026-10-29):** rows 17–25 (3 very small UCI categorical
sets, 1 census set, 2 document-word sets, 2 locally generated synthetic sets
with generator script `data_new/generate_synthetic.py`, and Online Retail II).
Rows 1–16 were backfilled with licence, access-date, owner/custodian, and
transformation columns on the same date. No licence is guessed anywhere in
this file: mirrors and UCI pages that do not state a licence are marked
unknown. Access date for all 25 datasets is 2026-10-29 (the original 16 were
also downloaded that day). `OnlineRetail.zip` (23,715,478 bytes)
is retained in the same folder as the provenance archive for `online_retail.dat`
and is **not** counted as a dataset.

## Datasets

| # | Dataset | File | Source URL | Size (bytes) | N transactions | M unique items | Avg length | Density % | Format notes  | Licence | Access date | Owner / custodian | Transformations |
|---|---------|------|------------|--------------|----------------|----------------|------------|-----------|-------------- |---|---|---|---|
| 1 | Groceries | groceries.csv | https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv | 500,843 | 9,835 | 169 | 4.41 | 2.6091 | CSV, comma-separated item names, one transaction per line, no header. One of the 5 original repo datasets.  | unknown — public mirror, licence not stated | 2026-10-29 | “Machine Learning with R” datasets (stedy mirror) | none |
| 2 | Chess | chess.dat | https://raw.githubusercontent.com/SarraLKSC/Minning-patterns-in-data/master/chess.dat | 342,294 | 3,196 | 75 | 37.00 | 49.3333 | Space-separated integer items, one transaction per line. Dense. One of the 5 original repo datasets.  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository (King-Rook vs King-Pawn, UCI origin) | none |
| 3 | Retail | retail.dat | https://raw.githubusercontent.com/ihewro/Apriori/master/retail.dat | 4,167,490 | 88,162 | 16,470 | 10.31 | 0.0626 | Space-separated integer items. Sparse. One of the 5 original repo datasets.  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository (Belgian retail) | none |
| 4 | Market basket | market_basket.csv | https://raw.githubusercontent.com/sharmaroshan/Market-Basket-Analysis/master/Market_Basket_Optimisation.csv | 302,908 | 7,501 | 119 | 3.91 | 3.2895 | CSV, comma-separated item names, one transaction per line, no header. One of the 5 original repo datasets.  | unknown — public mirror, licence not stated | 2026-10-29 | sharmaroshan mirror (Market Basket Optimisation) | none |
| 5 | Mushroom | mushroom.dat | https://raw.githubusercontent.com/hyperair/fptree/master/datasets/mushroom.dat | 570,408 | 8,124 | 119 | 23.00 | 19.3277 | Space-separated integer items. Dense. One of the 5 original repo datasets.  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository / UCI (mushroom) | none |
| 6 | Connect | connect.dat | https://raw.githubusercontent.com/Bushrahhh/apriori-algo-comparison/main/datasets/connect.dat | 9,255,309 | 67,557 | 129 | 43.00 | 33.3333 | Space-separated integer items. Dense (Connect-4 board positions).  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository (Connect-4, UCI origin) | none |
| 7 | Accidents | accidents.dat | https://raw.githubusercontent.com/HamiciRyadh/data-mining-close/main/datasets/accidents.dat | 35,509,834 | 340,183 | 468 | 33.81 | 7.2239 | Space-separated integer items. NOTE: this mirror file carries a first header line `340183 468` (N M); it was excluded from the transaction count. Largest file in the set (35.5 MB, under the 50 MB cap).  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository (Belgian traffic accidents) | first header line `340183 468` (N M) excluded from counts |
| 8 | Pumsb | pumsb.dat | https://media.githubusercontent.com/media/kafkasl/golang-ads/master/data/pumsb.dat | 16,689,761 | 49,046 | 2,113 | 74.00 | 3.5021 | Space-separated integer items, 0-based. Dense census data. NOTE: the file is Git-LFS-backed in its repo; the raw.githubusercontent URL returns only the LFS pointer, so the media.githubusercontent URL above is the working one.  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository (US Census PUMS) | none |
| 9 | Pumsb_star | pumsb_star.dat | https://media.githubusercontent.com/media/kafkasl/golang-ads/master/data/pumsb_star.dat | 11,291,914 | 49,046 | 2,088 | 50.48 | 2.4177 | Space-separated integer items, 0-based. Sparser variant of pumsb. Same LFS note as pumsb.dat.  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository (US Census PUMS variant) | none |
| 10 | Kosarak (FIMI) | kosarak.dat | https://media.githubusercontent.com/media/caph1993/mle-of-randomized-response/main/mle_vs_inv_python/data/kosarak.dat.txt | 32,029,467 | 990,002 | 41,270 | 8.10 | 0.0196 | Space-separated integer items. Click-stream data. Same LFS note as pumsb.dat. Largest transaction count in the set.  | unknown — public mirror, licence not stated | 2026-10-29 | FIMI repository (Hungarian news-portal clickstream) | none |
| 11 | Kosarak (SPMF sequence version) | Kosarak.txt | https://raw.githubusercontent.com/NguyenTriBaoThang/LC-FD-SPMSketch/main/fullydataset/Kosarak.txt | 11,057,644 | 187,140 | 28,252 | 8.03 | 0.0284 | SPMF sequence format: items separated by `-1`, each line terminated by `-2`. This is a DIFFERENT, smaller file from kosarak.dat (187,140 sequences vs 990,002 transactions) — kept as a separate dataset, not a duplicate. Strip `-1`/`-2` tokens to get plain itemsets.  | unknown — public mirror, licence not stated | 2026-10-29 | SPMF sequence mirror (NguyenTriBaoThang) | none as stored; strip `-1`/`-2` tokens when parsing |
| 12 | BMS-POS | BMS-POS.csv | https://raw.githubusercontent.com/cpearce/HARM/master/datasets/BMS-POS.csv | 11,900,318 | 515,596 | 1,657 | 6.53 | 0.3941 | CSV, comma-separated integer items with trailing commas. Point-of-sale click-stream data.  | unknown — public mirror, licence not stated | 2026-10-29 | Blue Martini Software (KDD Cup 2000) | none (source CSV has trailing commas) |
| 13 | BMS-WebView-1 | Transactional_BMS-WebView-1.csv | https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_BMS-WebView-1.csv | 946,589 | 59,009 | 494 | 2.51 | 0.5075 | Tab-separated integer items, CRLF line endings. NOTE: measured 59,009 transactions — the commonly cited FIMI figure is 59,601, so this mirror's version differs by 592 transactions; the measured number is what is reported here.  | unknown — public mirror, licence not stated | 2026-10-29 | Blue Martini Software clickstream (u-aizu mirror) | none |
| 14 | BMS-WebView-2 | BMS-WebView-2.dat | https://raw.githubusercontent.com/NguyenTriBaoThang/LC-FD-SPMSketch/main/fullydataset/BMSWebView2.txt | 2,239,752 | 77,512 | 3,340 | 4.62 | 0.1384 | CONVERTED: source was SPMF sequence format (`-1` separators, `-2` terminator); converted to plain space-separated integer itemsets by dropping `-1`/`-2` tokens (one itemset per line). Converted file is what is stored; the original .txt was removed after conversion.  | unknown — public mirror, licence not stated | 2026-10-29 | Blue Martini Software clickstream (SPMF mirror) | CONVERTED from SPMF sequence format: dropped `-1`/`-2` tokens |
| 15 | T10I4D100K (synthetic, IBM generator style) | T10I4D100K.dat | https://media.githubusercontent.com/media/Alexdruso/data-science-stuff/main/data_mining/lab2/data/T10I4D100K.dat | 4,022,055 | 100,000 | 870 | 10.10 | 1.1612 | Space-separated integer items. Synthetic. Same LFS note as pumsb.dat.  | unknown — public mirror, licence not stated | 2026-10-29 | IBM Quest synthetic generator (mirror) | none |
| 16 | Online Retail | online_retail.dat | Converted from https://archive.ics.uci.edu/static/public/352/online+retail.zip (Online Retail.xlsx) | 3,171,181 | 20,728 | 3,942 | 25.12 | 0.6373 | CONVERTED: UCI Online Retail II workbook grouped by InvoiceNo, items = StockCode. Cancelled invoices (InvoiceNo starting with `C`), non-positive quantities, and rows with missing StockCode were dropped. The source zip is retained as `OnlineRetail.zip`. Items are string StockCodes (e.g. `85123A`), not integers.  | unknown — public mirror, licence not stated | 2026-10-29 | UCI ML Repository (Online Retail) | CONVERTED: workbook grouped by InvoiceNo; cancelled invoices, non-positive quantities, missing StockCode dropped |
| 17 | Tic-tac-toe endgame | tic_tac_toe.dat | https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data | 54,606 | 958 | 29 | 10.00 | 34.4828 | Converted from UCI tic-tac-toe.data (CSV, 9 board cells + class). | unknown — licence not stated on the download page (not verified) | 2026-10-29 | UCI ML Repository | Each row converted to attribute=value items (9 cells + class item) |
| 18 | Car evaluation | car_evaluation.dat | https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data | 88,155 | 1,728 | 25 | 7.00 | 28.0000 | Converted from UCI car.data (CSV, 6 features + class). | unknown — licence not stated on the download page (not verified) | 2026-10-29 | UCI ML Repository | Each row converted to attribute=value items (6 features + class item) |
| 19 | Balance scale | balance_scale.dat | https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data | 15,625 | 625 | 23 | 5.00 | 21.7391 | Converted from UCI balance-scale.data (CSV, 4 features + class). | unknown — licence not stated on the download page (not verified) | 2026-10-29 | UCI ML Repository | Each row converted to attribute=value items (4 features + class item) |
| 20 | Adult census | adult_census.dat | https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data | 5,244,424 | 32,561 | 224 | 13.87 | 6.1918 | Converted from UCI adult.data (CSV, 14 attributes + income class). | unknown — licence not stated on the download page (not verified) | 2026-10-29 | UCI ML Repository (US Census Bureau source data) | fnlwgt column dropped (sampling weight, not an attribute); age bucketed by decade; hours-per-week bucketed (<30 / 30-50 / >50); capital gain/loss binarized (zero/positive); '?' values dropped; income class included as an item |
| 21 | NIPS papers (bag of words) | docword_nips.dat | https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz | 4,546,057 | 1,500 | 12,375 | 497.54 | 4.0206 | Converted from UCI docword.nips.txt (docword triples, gzip). | unknown — licence not stated on the download page (not verified) | 2026-10-29 | UCI ML Repository (bag-of-words collection) | One transaction per document; word IDs as items; word counts ignored; 44 of the 12,419 declared words never occur (measured M = 12,375) |
| 22 | Enron emails (bag of words) | docword_enron.dat | https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz | 24,464,283 | 39,861 | 28,099 | 93.08 | 0.3313 | Converted from UCI docword.enron.txt (docword triples, gzip). | unknown — licence not stated on the download page (not verified) | 2026-10-29 | UCI ML Repository (bag-of-words collection) | One transaction per document; word IDs as items; word counts ignored; 3 of the 28,102 declared words never occur (measured M = 28,099) |
| 23 | Synthetic sparse 2000 (GENERATED) | synthetic_sparse_2000.dat | Generated locally — no download URL (see generate_synthetic.py) | 42,623 | 2,000 | 499 | 7.42 | 1.4864 | GENERATED, not downloaded: Zipf-weighted items, Poisson(8) lengths. | N/A — generated by this project | 2026-10-29 | This project (generated locally) | None — generated; generator script data_new/generate_synthetic.py, seed 42, planted itemset {1,2,3} in 20% of transactions; item 500 never drawn (measured M = 499) |
| 24 | Synthetic dense 5000 (GENERATED) | synthetic_dense_5000.dat | Generated locally — no download URL (see generate_synthetic.py) | 428,406 | 5,000 | 50 | 30.56 | 61.1236 | GENERATED, not downloaded: uniform items, Normal(30,5) lengths. | N/A — generated by this project | 2026-10-29 | This project (generated locally) | None — generated; generator script data_new/generate_synthetic.py, seed 42, planted itemsets {1,2} and {3,4,5} in 30% of transactions each |
| 25 | Online Retail II | online_retail_ii.dat | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip | 3,069,670 | 22,103 | 4,317 | 22.62 | 0.5239 | Converted from UCI online_retail_II.xlsx (2009-2011, 525,461 source rows). | unknown — licence not stated on the download page (not verified) | 2026-10-29 | UCI ML Repository (Online Retail II) | Grouped by Invoice; cancelled invoices, non-positive quantities, missing StockCode dropped; source zip (45,622,418 bytes) kept only in scratch, not in data_new |

## Coverage by type

- Sparse retail: groceries, market_basket, retail, BMS-POS, online_retail (5)
- Dense: chess, mushroom, connect, accidents, pumsb, pumsb_star (6)
- Click-stream: kosarak (FIMI), Kosarak (SPMF sequence), BMS-WebView-1, BMS-WebView-2 (4)
- Synthetic (IBM generator style): T10I4D100K (1)
- Very small categorical (UCI, rows as attribute=value transactions): tic-tac-toe, car evaluation, balance scale (3)
- Census: adult census, pumsb, pumsb_star (3, adult added in extension)
- Document-word (one transaction per document): NIPS papers, Enron emails (2)
- Locally generated synthetic (seed 42): sparse 2000, dense 5000 (2)
- Additional retail: Online Retail II (1)

## Failed / skipped downloads (honest record)

- **T40I10D100K** — NOT downloaded. The FIMI hosts
  (`fimi.uantwerpen.be`, `fimi.ua.ac.be`) are unreachable from this machine
  (connection failures, HTTP 000, after one retry each). GitHub code search
  found only derived outputs (`.fim` frequent-itemset files, a frequency
  table, metadata) — no repository hosting the raw transaction file — and the
  u-aizu.ac.jp transactional-database directory does not carry it either
  (its T40 URL returns an HTML error page). Skipped rather than substituted
  with a generated file, since the task asked for downloads.
- **Foodmart** — NOT downloaded. The only GitHub hit for `foodmart.dat` was an
  SSAS database file, not a transactional dataset; no public transactional
  mirror was found.
- **SPMF direct links** (`philippe-fournier-viger.com/spmf/datasets/*.txt`) —
  return HTTP 300 "Multiple Choices" pages; the `datasets.php/*.txt` variants
  return an HTML landing page, not data. Not usable.
- **FIMI direct links** (`fimi.uantwerpen.be/data/*.dat`) — host unreachable
  (see T40 note). All FIMI-origin datasets above came from GitHub mirrors
  instead, and their measured statistics match the published FIMI figures
  (accidents 340,183 / chess 3,196 / connect 67,557 / mushroom 8,124 /
  retail 88,162 / kosarak 990,002 / pumsb 49,046 / BMS-WebView-2 77,512),
  except BMS-WebView-1 (see its note).
- **Online Retail CSV mirrors** (Shailaja-Gupta, hellich, CS133, evanvolgas
  sample) — either LFS pointers whose objects are missing from the media
  endpoint, a 404, or a 1,000-row sample. The full UCI workbook download was
  used instead (see dataset 16).
- **Files over the 50 MB cap**: none of the accepted datasets exceed it
  (largest is accidents.dat at 35.5 MB). Nothing was skipped for size.

- **Ta-Feng grocery** — NOT downloaded in the extension. Its only sources
  found are Kaggle (login-gated, excluded by the no-login rule) and the
  choice-learn package (which itself loads it from Kaggle). GitHub repo and
  code search found no public raw transaction mirror. The very-big slot it
  was meant to fill is already covered by kosarak (990,002) and BMS-POS
  (515,596); Online Retail II was added instead as a distinct retail set
  but measured only 22,103 transactions from 525,461 source rows, so it is
  NOT a very-big set and is not claimed as one.

## Method

Downloaded with `curl -L --retry 1` and a browser User-Agent. Every file was
validated by parsing (not by trusting the source's claimed size): a file was
only accepted if it parsed into transactions of items. HTML error pages
(SPMF, u-aizu T40) and Git-LFS pointer files were detected and rejected or
re-fetched via the media endpoint as noted. Statistics were computed with a
single Python pass over each file (see `/tmp/convert_and_stats.py`, scratch
only, not part of the deliverable).
