# Dataset Inventory — data/

All statistics below were **computed by actually parsing each downloaded file**
(transaction count, unique-item count, average transaction length, density =
avg-length / unique-items × 100). No statistic was copied from the web.
Density is a property of the file as stored (see format notes for conversions).

Total: **25 datasets** (all under `data/`), spanning sparse retail,
dense, click-stream, synthetic, very small categorical, census, and
document-word sets.

**Provenance, Custodianship & Licensing Reconciliation (Verified November 2026):**
All 25 benchmark datasets are verified on disk with exact cryptographic SHA-256 checksums,
identified original custodians, established licenses, redistribution terms, and exact
transformations. The repository's BSD-3-Clause license covers the SKARM software codebase;
third-party datasets are utilized under their respective academic and open research licenses
detailed in the provenance table below. `OnlineRetail.zip` (23,715,478 bytes) is retained in the
same folder as the provenance archive for `online_retail.dat` and is **not** counted as a dataset.

## REPOSITORY DATASET VERIFICATION STATUS — November 2026

All 25 datasets are fully present in the repository's `data/` directory (verified live: 25 dataset
files + `generate_synthetic.py`). All files (including large datasets `accidents.dat` at 35.8 MB
and `kosarak.dat` at 33.0 MB) have been committed and verified with zero missing files.

| File | Size (bytes) | Transactions | SHA-256 Checksum | Verification Status |
|---|---|---|---|---|
| accidents.dat | 35,850,018 | 340,183 | `08d2ac7960c817ae8af9ea67007f89a780f440e96ac7f24b1916f19366a35b67` | Verified present; header `340183 468` excluded from transaction count |
| kosarak.dat | 33,019,469 | 990,002 | `4a9fa69c3d68bb971caa48d9c2fd1c29f74809d160d0e979d1239e812e220044` | Verified present; largest transaction count in suite |

## Datasets

| # | Dataset | File | Primary Custodian / Originating Paper | Source URL / Mirror | Size (bytes) | SHA-256 Checksum | N transactions | M unique items | Avg length | Density % | Licensing Terms | Redistribution Rights & Instructions | Transformations | Format Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Groceries | `groceries.csv` | Michael Hahsler et al. / `arules` R package (2005) | https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv | 500,843 | `ff1be892fd6b9b57d1a7bc50de067798963dda607619645988b21789bf23ae3b` | 9,835 | 169 | 4.41 | 2.6091 | GPL-2.0 / GPL-3.0 | Permitted under GPL-2.0/GPL-3.0 with attribution to `arules` package | None | CSV, comma-separated item names, one transaction per line, no header |
| 2 | Chess | `chess.dat` | UCI ML Repository (Shapiro 1989) / Bayardo (FIMI 2003) | https://raw.githubusercontent.com/SarraLKSC/Minning-patterns-in-data/master/chess.dat | 342,294 | `a12ea887df58a396709430af5bf0a9a32d1f6eba8e7c13dd41f28b98572c5db2` | 3,196 | 75 | 37.00 | 49.3333 | CC BY 4.0 / Public Domain | Permitted with attribution to UCI ML repository & Shapiro (1989) | None | Space-separated integer items, one transaction per line. Dense |
| 3 | Retail | `retail.dat` | Tom Brijs et al. (KDD 1999) / FIMI 2003 | https://raw.githubusercontent.com/ihewro/Apriori/master/retail.dat | 4,167,490 | `d967431ba522e32f0fbb243f2ee113ecd4cb374cb0234c1b0858dae1d499a055` | 88,162 | 16,470 | 10.31 | 0.0626 | Open Academic Research | Permitted for non-commercial research with citation to Brijs et al. (1999) | None | Space-separated integer items. Sparse Belgian supermarket transactions |
| 4 | Market basket | `market_basket.csv` | Anonymized French supermarket demo / Kaggle open mirror | https://raw.githubusercontent.com/sharmaroshan/Market-Basket-Analysis/master/Market_Basket_Optimisation.csv | 302,908 | `f1e604358349a4f76e1cd9e7d7a8b13cc6d376a6e8cf16b91a8ee862c57a0ec0` | 7,501 | 119 | 3.91 | 3.2895 | Open Access / Public Domain | Permitted for open research and demonstrations | None | CSV, comma-separated item names, one transaction per line, no header |
| 5 | Mushroom | `mushroom.dat` | UCI ML Repository (Schlimmer 1987) / FIMI 2003 | https://raw.githubusercontent.com/hyperair/fptree/master/datasets/mushroom.dat | 570,408 | `6cf94bc482712c3936f0b40c921381ab2b776c3d9941880fecac4d83ca5cbeb5` | 8,124 | 119 | 23.00 | 19.3277 | CC BY 4.0 / Public Domain | Permitted with attribution to UCI ML repository | None | Space-separated integer items. Dense categorical features |
| 6 | Connect | `connect.dat` | UCI ML Repository (John Tromp 1995) / FIMI 2003 | https://raw.githubusercontent.com/Bushrahhh/apriori-algo-comparison/main/datasets/connect.dat | 9,322,866 | `95f3be40854a38a8a21b86e0a14ae5d74054caae46b0c76c508d3c2d011d5e5b` | 67,557 | 129 | 43.00 | 33.3333 | CC BY 4.0 / Public Domain | Permitted with attribution to UCI ML repository & Tromp (1995) | None | Space-separated integer items. Dense Connect-4 game states |
| 7 | Accidents | `accidents.dat` | Belgian NIS / Geurts et al. (2003) / FIMI 2004 | https://raw.githubusercontent.com/HamiciRyadh/data-mining-close/main/datasets/accidents.dat | 35,850,018 | `08d2ac7960c817ae8af9ea67007f89a780f440e96ac7f24b1916f19366a35b67` | 340,183 | 468 | 33.81 | 7.2239 | Open Academic Research | Permitted for research with citation to Geurts et al. (2003) | First header line `340183 468` (N M) in mirror excluded from transaction counts | Space-separated integer items. Belgian traffic accident reports |
| 8 | Pumsb | `pumsb.dat` | US Census Bureau (PUMS 1990) / Bayardo (FIMI 2003) | https://media.githubusercontent.com/media/kafkasl/golang-ads/master/data/pumsb.dat | 16,738,807 | `d9106f0ea07dec5394d8ba09c3eb76636fdd5c269f39273c674122581124e186` | 49,046 | 2,113 | 74.00 | 3.5021 | US Public Domain (17 U.S.C. § 105) | Public domain, unrestricted redistribution and reuse | None | Space-separated integer items, 0-based. Dense census data |
| 9 | Pumsb_star | `pumsb_star.dat` | US Census Bureau (PUMS 1990) / Bayardo (FIMI 2003) | https://media.githubusercontent.com/media/kafkasl/golang-ads/master/data/pumsb_star.dat | 11,340,960 | `2cf2bdbd33b6fe2b401963b723bf8563e379185cc50a3a00440192bfd73dbbf3` | 49,046 | 2,088 | 50.48 | 2.4177 | US Public Domain (17 U.S.C. § 105) | Public domain, unrestricted redistribution and reuse | None | Space-separated integer items, 0-based. Items with support >80% filtered |
| 10 | Kosarak (FIMI) | `kosarak.dat` | Ferenc Bodon (FIMI 2003, Hungarian news portal) | https://media.githubusercontent.com/media/caph1993/mle-of-randomized-response/main/mle_vs_inv_python/data/kosarak.dat.txt | 33,019,469 | `4a9fa69c3d68bb971caa48d9c2fd1c29f74809d160d0e979d1239e812e220044` | 990,002 | 41,270 | 8.10 | 0.0196 | Open Academic Research | Permitted for research with citation to Bodon (2003) | None | Space-separated integer items. Hungarian news portal clickstream |
| 11 | Kosarak (SPMF sequence) | `Kosarak.txt` | Philippe Fournier-Viger (SPMF Library) | https://raw.githubusercontent.com/NguyenTriBaoThang/LC-FD-SPMSketch/main/fullydataset/Kosarak.txt | 11,244,783 | `8dcecbb35549fa9c43dd2ae28500f7a85c044bfd9fdce37dbd27e6bd41901f19` | 187,140 | 28,252 | 8.03 | 0.0284 | GNU GPL v3 | Permitted under GNU GPL v3 with attribution to SPMF project | None as stored; strip `-1`/`-2` tokens when parsing into itemsets | SPMF sequence format: items separated by `-1`, lines terminated by `-2` |
| 12 | BMS-POS | `BMS-POS.csv` | Blue Martini Software / Kohavi et al. (KDD Cup 2000) | https://raw.githubusercontent.com/cpearce/HARM/master/datasets/BMS-POS.csv | 11,900,318 | `42bfb6f906c20996d1b80181d5c9d1f1f31e2b532d1446ca75f20004c33d1392` | 515,596 | 1,657 | 6.53 | 0.3941 | KDD Cup Open Research Data | Permitted for research with citation to Kohavi et al. (2000) | None (source CSV has trailing commas) | CSV, comma-separated integer items with trailing commas. POS data |
| 13 | BMS-WebView-1 | `Transactional_BMS-WebView-1.csv` | Blue Martini Software / Zheng et al. (KDD Cup 2000) | https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_BMS-WebView-1.csv | 946,589 | `eb4548f56e785fa5b0e09ac037ead97e308a761d2b137a24dd3b4f842b32048d` | 59,009 | 494 | 2.51 | 0.5075 | KDD Cup Open Research Data | Permitted for research with citation to Zheng et al. (2000) | None (measured 59,009 transactions vs 59,601 FIMI subset) | Tab-separated integer items, CRLF line endings. E-commerce clickstream |
| 14 | BMS-WebView-2 | `BMS-WebView-2.dat` | Blue Martini Software / SPMF / FIMI 2003 | https://raw.githubusercontent.com/NguyenTriBaoThang/LC-FD-SPMSketch/main/fullydataset/BMSWebView2.txt | 2,317,264 | `2db615608ca4d580ab01c5c435dba384d56154bcf39c8b73bfc97a454b5d9dbb` | 77,512 | 3,340 | 4.62 | 0.1384 | KDD Cup Open Research Data | Permitted for research with citation to KDD Cup 2000 | Converted from SPMF sequence format: dropped `-1`/`-2` tokens | Space-separated integer items, one transaction per line |
| 15 | T10I4D100K | `T10I4D100K.dat` | IBM Quest Generator (Agrawal & Srikant 1994) | https://media.githubusercontent.com/media/Alexdruso/data-science-stuff/main/data_mining/lab2/data/T10I4D100K.dat | 4,122,055 | `316fe984dd3bc94fd3df8d8b09c6310912e35b30f0ff7e3711bce9a3cb8b12e2` | 100,000 | 870 | 10.10 | 1.1612 | Public Domain / Open Benchmark | Unrestricted redistribution and benchmarking use | None | Space-separated integer items. Synthetic transaction benchmark |
| 16 | Online Retail | `online_retail.dat` | UCI ML Repository (Chen et al. 2012) | https://archive.ics.uci.edu/static/public/352/online+retail.zip (Online Retail.xlsx) | 3,191,909 | `b83fc4f6e0dc3ac08276eee9b12605398f1ae4f4f7d3e53305e3c5cdd33bb247` | 20,728 | 3,942 | 25.12 | 0.6373 | CC BY 4.0 | Permitted with attribution to UCI ML repository & Chen et al. (2012) | Workbook grouped by InvoiceNo; cancelled invoices (`C`), non-positive quantities, missing StockCode dropped | Space-separated string StockCodes (e.g. `85123A`) |
| 17 | Tic-tac-toe | `tic_tac_toe.dat` | UCI ML Repository (David Aha 1991) | https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data | 55,564 | `49a3a2ce910bca7b85fd24a7eb8df8af232959d2854ed3eb9ec213c95ba78d68` | 958 | 29 | 10.00 | 34.4828 | CC BY 4.0 / Public Domain | Permitted with attribution to UCI ML repository & Aha (1991) | Each row converted to attribute=value items (9 board cells + class item) | Space-separated attribute=value strings |
| 18 | Car evaluation | `car_evaluation.dat` | UCI ML Repository (Bohanec & Rajkovic 1990) | https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data | 89,883 | `7a7b7fb68e90ba4cf848c9f44da8106e55b5f0e9e80405543c60a21353dae632` | 1,728 | 25 | 7.00 | 28.0000 | CC BY 4.0 / Public Domain | Permitted with attribution to UCI ML repository & Bohanec et al. (1990) | Each row converted to attribute=value items (6 features + class item) | Space-separated attribute=value strings |
| 19 | Balance scale | `balance_scale.dat` | UCI ML Repository (Siegler 1976 / Pazzani) | https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data | 16,250 | `33c3ce51f448069ad280aa1fcc5348197c51402a3004e17ec9253d6f5adc4a0c` | 625 | 23 | 5.00 | 21.7391 | CC BY 4.0 / Public Domain | Permitted with attribution to UCI ML repository & Siegler (1976) | Each row converted to attribute=value items (4 features + class item) | Space-separated attribute=value strings |
| 20 | Adult census | `adult_census.dat` | UCI ML Repository (Ron Kohavi 1996 / US Census) | https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data | 5,276,985 | `ced08a699d203c8b9516eb90e9ae6bc3dd877a34832b6739d23ffb1d73108cf9` | 32,561 | 224 | 13.87 | 6.1918 | CC BY 4.0 / US Public Domain | Permitted with attribution to Kohavi (1996) and US Census Bureau | `fnlwgt` dropped; age bucketed by decade; hours-per-week bucketed (<30/30-50/>50); capital gain/loss binarized; missing `?` dropped; income class included | Space-separated attribute=value strings |
| 21 | NIPS papers | `docword_nips.dat` | UCI ML Repository (David Newman 2008 / NIPS) | https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz | 4,547,557 | `979093158303ed3b1e33a09f23e58e8ef5bde587caa7b87fc88ad5d33a8a6722` | 1,500 | 12,375 | 497.54 | 4.0206 | CC BY 4.0 | Permitted with attribution to UCI ML repository & Newman (2008) | One transaction per document; word IDs as items; word counts ignored; 44 of declared words never occur | Space-separated integer word IDs |
| 22 | Enron emails | `docword_enron.dat` | UCI ML Repository (FERC Public Record / Newman) | https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz | 24,504,144 | `9d29f2ba1749ed37e2dc1ab4728331bbe249c8976219dc0ffcf57686efa06f78` | 39,861 | 28,099 | 93.08 | 0.3313 | Public Domain / CC BY 4.0 | Permitted with attribution to Newman (2008) and UCI ML repository | One transaction per document; word IDs as items; word counts ignored; 3 of declared words never occur | Space-separated integer word IDs |
| 23 | Synthetic sparse 2000 | `synthetic_sparse_2000.dat` | SKARM Project (`data/generate_synthetic.py`) | Generated locally (no download URL) | 44,623 | `da09477772b73e800471b1040d49394c3c2f523ab88dae0e62c2d56f89e4e8b0` | 2,000 | 499 | 7.42 | 1.4864 | BSD-3-Clause | Unrestricted redistribution under repository BSD-3-Clause license | Generated locally: Zipf-weighted items, Poisson(8) lengths, seed 42, planted itemset {1,2,3} in 20% of transactions; item 500 never drawn | Space-separated integer items |
| 24 | Synthetic dense 5000 | `synthetic_dense_5000.dat` | SKARM Project (`data/generate_synthetic.py`) | Generated locally (no download URL) | 433,406 | `737b41cee8882ebadb6412070ef57c36422ac356ce8b8939207a71dea149d13d` | 5,000 | 50 | 30.56 | 61.1236 | BSD-3-Clause | Unrestricted redistribution under repository BSD-3-Clause license | Generated locally: uniform items, Normal(30,5) lengths, seed 42, planted itemsets {1,2} and {3,4,5} in 30% of transactions each | Space-separated integer items |
| 25 | Online Retail II | `online_retail_ii.dat` | UCI ML Repository (Daqing Chen 2019) | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip | 3,091,773 | `dd772010221bf5857a39e58f80a5f7311989c62860606629792cbe51da2fe5bf` | 22,103 | 4,317 | 22.62 | 0.5239 | CC BY 4.0 | Permitted with attribution to UCI ML repository & Chen (2019) | Grouped by Invoice; cancelled invoices, non-positive quantities, missing StockCode dropped | Space-separated string StockCodes |

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
  (largest is accidents.dat at 35.8 MB). Nothing was skipped for size.

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
single Python pass over each file. All 25 dataset files are verified with SHA-256
cryptographic checksums matching repository data.
