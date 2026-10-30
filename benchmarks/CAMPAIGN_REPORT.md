# skarm benchmark campaign report (PC run)

> COMPLETE — all datasets in thresholds.json have a checkpoint file.

## Environment

- platform: AMD64, Windows 11
- cpu_count: 16
- workers: 12
- run_mode: pc-parallel (one worker per dataset)
- python_version: 3.12.10
- scikit_learn_version: 1.9.1
- mlxtend_version: 0.25.0
- numpy_version: 2.5.4
- pandas_version: 3.1.0
- scipy_version: 1.18.1
- trial_timeout_s: 600
- rule params: {'metric': 'confidence', 'min_threshold': 0.25}
- estimator params: {'max_candidates': 2000000, 'sparse_output': True, 'on_candidate_explosion': 'degrade'}
- memory method: peak RSS sampled every 10 ms from /proc/self/status VmRSS, minus pre-trial baseline RSS (0.0 where /proc is unavailable)
- runtime method: time.perf_counter around AssociationRulesEstimator.fit on a pre-encoded matrix (encoding time reported separately per dataset); mlxtend runtime covers TransactionEncoder-equivalent dense DataFrame build + apriori + association_rules

## Per-cell results (seconds; mean / p50 / p95 over OK trials only)

| dataset | threshold | engine | ok trials | mean | p50 | p95 | mean peak MB | itemsets | rules | degraded |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| accidents | 0.5 | apriori | 0/5 | 0 | 0.0 | 0.0 | 0 | - | - | - |
| accidents | 0.5 | eclat | 5/5 | 288.156 | 273.564 | 333.357 | 1738.2 | 8057 | 375774 | False |
| accidents | 0.5 | fpgrowth | 5/5 | 156.528 | 157.075 | 181.314 | 955.1 | 8057 | 375774 | False |
| accidents | 0.5 | mlxtend | 5/5 | 22.064 | 21.913 | 22.809 | 5494.2 | 8057 | 375774 | - |
| accidents | 0.6 | apriori | 5/5 | 473.543 | 475.249 | 478.175 | 1860.6 | 2074 | 54170 | False |
| accidents | 0.6 | eclat | 5/5 | 97.928 | 97.648 | 100.266 | 1351.0 | 2074 | 54170 | False |
| accidents | 0.6 | fpgrowth | 5/5 | 60.916 | 61.306 | 62.38 | 90.4 | 2074 | 54170 | False |
| accidents | 0.6 | mlxtend | 5/5 | 4.761 | 4.753 | 4.999 | 1470.8 | 2074 | 54170 | - |
| adult_census | 0.05 | apriori | 5/5 | 76.623 | 70.041 | 105.091 | 957.5 | 11390 | 376953 | False |
| adult_census | 0.05 | eclat | 5/5 | 20.95 | 20.337 | 24.008 | 887.9 | 11390 | 376953 | False |
| adult_census | 0.05 | fpgrowth | 5/5 | 19.044 | 18.844 | 19.827 | 950.9 | 11390 | 376953 | False |
| adult_census | 0.05 | mlxtend | 5/5 | 19.138 | 19.06 | 20.088 | 717.0 | 11390 | 376953 | - |
| adult_census | 0.1 | apriori | 5/5 | 23.358 | 20.406 | 31.181 | 160.6 | 3216 | 78167 | False |
| adult_census | 0.1 | eclat | 5/5 | 5.37 | 5.362 | 5.443 | 142.1 | 3216 | 78167 | False |
| adult_census | 0.1 | fpgrowth | 5/5 | 5.532 | 5.551 | 5.579 | 150.6 | 3216 | 78167 | False |
| adult_census | 0.1 | mlxtend | 5/5 | 3.478 | 3.482 | 3.548 | 208.8 | 3216 | 78167 | - |
| balance_scale | 0.2 | apriori | 5/5 | 0.033 | 0.032 | 0.039 | 0.1 | 22 | 0 | False |
| balance_scale | 0.2 | eclat | 5/5 | 0.009 | 0.008 | 0.012 | 0.2 | 22 | 0 | False |
| balance_scale | 0.2 | fpgrowth | 5/5 | 0.032 | 0.031 | 0.033 | 0.1 | 22 | 0 | False |
| balance_scale | 0.2 | mlxtend | 5/5 | 0.006 | 0.006 | 0.006 | 0.0 | 22 | 0 | - |
| balance_scale | 0.3 | apriori | 5/5 | 0.022 | 0.021 | 0.024 | 0.3 | 2 | 0 | False |
| balance_scale | 0.3 | eclat | 5/5 | 0.008 | 0.008 | 0.009 | 0.0 | 2 | 0 | False |
| balance_scale | 0.3 | fpgrowth | 5/5 | 0.022 | 0.022 | 0.024 | 0.0 | 2 | 0 | False |
| balance_scale | 0.3 | mlxtend | 5/5 | 0.017 | 0.006 | 0.062 | 0.2 | 2 | 0 | - |
| bms_pos | 0.02 | apriori | 30/30 | 227.205 | 232.565 | 237.223 | 22.8 | 308 | 489 | False |
| bms_pos | 0.02 | eclat | 30/30 | 12.529 | 12.675 | 13.686 | 39.5 | 308 | 489 | False |
| bms_pos | 0.02 | fpgrowth | 30/30 | 94.078 | 94.099 | 96.577 | 107.5 | 308 | 489 | False |
| bms_pos | 0.02 | mlxtend | 5/5 | 27.43 | 32.487 | 32.891 | 164.5 | 308 | 489 | - |
| bms_webview_1 | 0.005 | apriori | 5/5 | 31.145 | 29.175 | 44.215 | 10.0 | 201 | 59 | False |
| bms_webview_1 | 0.005 | eclat | 5/5 | 0.86 | 0.875 | 0.985 | 8.5 | 201 | 59 | False |
| bms_webview_1 | 0.005 | fpgrowth | 5/5 | 5.312 | 5.243 | 6.58 | 2.9 | 201 | 59 | False |
| bms_webview_1 | 0.005 | mlxtend | 5/5 | 2.677 | 2.507 | 3.622 | 1895.9 | 201 | 59 | - |
| bms_webview_1 | 0.01 | apriori | 5/5 | 6.853 | 6.851 | 6.938 | 14.7 | 77 | 17 | False |
| bms_webview_1 | 0.01 | eclat | 5/5 | 0.377 | 0.379 | 0.388 | 8.2 | 77 | 17 | False |
| bms_webview_1 | 0.01 | fpgrowth | 5/5 | 2.295 | 2.294 | 2.357 | 2.5 | 77 | 17 | False |
| bms_webview_1 | 0.01 | mlxtend | 5/5 | 0.397 | 0.409 | 0.411 | 401.3 | 77 | 17 | - |
| bms_webview_2 | 0.005 | apriori | 5/5 | 64.299 | 63.043 | 70.419 | 14.6 | 408 | 877 | False |
| bms_webview_2 | 0.005 | eclat | 5/5 | 1.09 | 1.075 | 1.161 | 9.2 | 408 | 877 | False |
| bms_webview_2 | 0.005 | fpgrowth | 5/5 | 7.658 | 6.592 | 10.757 | 2.4 | 408 | 877 | False |
| bms_webview_2 | 0.005 | mlxtend | 5/5 | 3.971 | 4.037 | 4.268 | 3423.4 | 408 | 877 | - |
| bms_webview_2 | 0.01 | apriori | 5/5 | 12.511 | 11.028 | 16.674 | 16.3 | 81 | 52 | False |
| bms_webview_2 | 0.01 | eclat | 5/5 | 0.947 | 0.849 | 1.338 | 2.7 | 81 | 52 | False |
| bms_webview_2 | 0.01 | fpgrowth | 5/5 | 4.791 | 4.139 | 7.348 | 1.7 | 81 | 52 | False |
| bms_webview_2 | 0.01 | mlxtend | 5/5 | 1.744 | 1.716 | 1.938 | 588.4 | 81 | 52 | - |
| car_evaluation | 0.2 | apriori | 5/5 | 0.087 | 0.087 | 0.088 | 0.1 | 31 | 16 | False |
| car_evaluation | 0.2 | eclat | 5/5 | 0.031 | 0.029 | 0.037 | 0.1 | 31 | 16 | False |
| car_evaluation | 0.2 | fpgrowth | 5/5 | 0.112 | 0.11 | 0.119 | 0.3 | 31 | 16 | False |
| car_evaluation | 0.2 | mlxtend | 5/5 | 0.011 | 0.011 | 0.012 | 0.0 | 31 | 16 | - |
| car_evaluation | 0.3 | apriori | 5/5 | 0.091 | 0.069 | 0.178 | 10.0 | 12 | 4 | False |
| car_evaluation | 0.3 | eclat | 5/5 | 0.026 | 0.026 | 0.027 | 0.1 | 12 | 4 | False |
| car_evaluation | 0.3 | fpgrowth | 5/5 | 0.073 | 0.071 | 0.079 | 0.0 | 12 | 4 | False |
| car_evaluation | 0.3 | mlxtend | 5/5 | 0.009 | 0.008 | 0.01 | 0.1 | 12 | 4 | - |
| chess | 0.85 | apriori | 30/30 | 5.762 | 5.675 | 6.964 | 167.4 | 2669 | 95482 | False |
| chess | 0.85 | eclat | 30/30 | 3.475 | 3.467 | 3.58 | 93.0 | 2669 | 95482 | False |
| chess | 0.85 | fpgrowth | 30/30 | 3.489 | 3.444 | 3.846 | 105.7 | 2669 | 95482 | False |
| chess | 0.85 | mlxtend | 5/5 | 3.495 | 3.52 | 3.566 | 19.9 | 2669 | 95482 | - |
| connect | 0.95 | apriori | 30/30 | 87.865 | 88.56 | 110.095 | 304.9 | 2201 | 78376 | False |
| connect | 0.95 | eclat | 30/30 | 14.223 | 14.217 | 15.299 | 191.0 | 2201 | 78376 | False |
| connect | 0.95 | fpgrowth | 30/30 | 10.049 | 9.723 | 13.768 | 41.0 | 2201 | 78376 | False |
| connect | 0.95 | mlxtend | 5/5 | 3.443 | 3.471 | 3.563 | 284.2 | 2201 | 78376 | - |
| docword_enron | 0.5 | apriori | 5/5 | 14.751 | 14.358 | 16.447 | 166.6 | 0 | 0 | False |
| docword_enron | 0.5 | eclat | 5/5 | 8.368 | 8.04 | 9.494 | 17.7 | 0 | 0 | False |
| docword_enron | 0.5 | fpgrowth | 5/5 | 0.446 | 0.445 | 0.455 | 7.6 | 0 | 0 | False |
| docword_enron | 0.5 | mlxtend | 0/5 | 0 | 0.0 | 0.0 | 0 | - | - | - |
| docword_enron | 0.7 | apriori | 5/5 | 14.254 | 14.242 | 14.359 | 191.1 | 0 | 0 | False |
| docword_enron | 0.7 | eclat | 5/5 | 7.955 | 7.96 | 7.991 | 17.7 | 0 | 0 | False |
| docword_enron | 0.7 | fpgrowth | 5/5 | 0.441 | 0.444 | 0.448 | 1.1 | 0 | 0 | False |
| docword_enron | 0.7 | mlxtend | 0/5 | 0 | 0.0 | 0.0 | 0 | - | - | - |
| docword_nips | 0.5 | apriori | 30/30 | 209.504 | 195.493 | 269.294 | 2672.8 | 31582 | 1148468 | False |
| docword_nips | 0.5 | eclat | 30/30 | 50.231 | 44.471 | 90.977 | 2345.5 | 31582 | 1148468 | False |
| docword_nips | 0.5 | fpgrowth | 30/30 | 81.746 | 81.308 | 85.885 | 2560.6 | 31582 | 1148468 | False |
| docword_nips | 0.5 | mlxtend | 5/5 | 53.907 | 49.349 | 75.601 | 1079.8 | 31582 | 1148468 | - |
| groceries | 0.01 | apriori | 30/30 | 2.177 | 2.155 | 2.335 | 1.6 | 333 | 170 | False |
| groceries | 0.01 | eclat | 30/30 | 0.166 | 0.16 | 0.203 | 0.6 | 333 | 170 | False |
| groceries | 0.01 | fpgrowth | 30/30 | 0.871 | 0.87 | 0.938 | 1.1 | 333 | 170 | False |
| groceries | 0.01 | mlxtend | 5/5 | 0.128 | 0.126 | 0.142 | 107.0 | 333 | 170 | - |
| groceries | 0.02 | apriori | 30/30 | 1.073 | 1.075 | 1.129 | 0.8 | 122 | 49 | False |
| groceries | 0.02 | eclat | 30/30 | 0.126 | 0.126 | 0.138 | 0.2 | 122 | 49 | False |
| groceries | 0.02 | fpgrowth | 30/30 | 0.686 | 0.679 | 0.762 | 0.8 | 122 | 49 | False |
| groceries | 0.02 | mlxtend | 5/5 | 0.055 | 0.052 | 0.066 | 33.3 | 122 | 49 | - |
| kosarak | 0.005 | apriori | 0/5 | 0 | 0.0 | 0.0 | 0 | - | - | - |
| kosarak | 0.005 | eclat | 5/5 | 28.303 | 28.23 | 28.732 | 349.2 | 1618 | 5890 | False |
| kosarak | 0.005 | fpgrowth | 5/5 | 473.859 | 477.521 | 481.244 | 69.3 | 1618 | 5890 | False |
| kosarak | 0.01 | apriori | 5/5 | 250.774 | 249.876 | 253.115 | 690.8 | 383 | 1037 | False |
| kosarak | 0.01 | eclat | 5/5 | 21.294 | 21.432 | 22.39 | 277.1 | 383 | 1037 | False |
| kosarak | 0.01 | fpgrowth | 5/5 | 198.894 | 199.307 | 203.446 | 76.0 | 383 | 1037 | False |
| kosarak_spmf | 0.01 | apriori | 5/5 | 50.472 | 50.581 | 51.339 | 95.8 | 377 | 999 | False |
| kosarak_spmf | 0.01 | eclat | 5/5 | 4.238 | 4.253 | 4.321 | 25.3 | 377 | 999 | False |
| kosarak_spmf | 0.01 | fpgrowth | 5/5 | 23.023 | 23.25 | 23.596 | 4.6 | 377 | 999 | False |
| market_basket | 0.02 | apriori | 5/5 | 0.688 | 0.694 | 0.71 | 0.0 | 103 | 28 | False |
| market_basket | 0.02 | eclat | 5/5 | 0.089 | 0.088 | 0.102 | 0.0 | 103 | 28 | False |
| market_basket | 0.02 | fpgrowth | 5/5 | 0.449 | 0.434 | 0.524 | 0.4 | 103 | 28 | False |
| market_basket | 0.02 | mlxtend | 5/5 | 0.036 | 0.036 | 0.039 | 23.7 | 103 | 28 | - |
| market_basket | 0.03 | apriori | 5/5 | 0.449 | 0.425 | 0.556 | 0.0 | 54 | 10 | False |
| market_basket | 0.03 | eclat | 5/5 | 0.076 | 0.075 | 0.085 | 0.4 | 54 | 10 | False |
| market_basket | 0.03 | fpgrowth | 5/5 | 0.37 | 0.368 | 0.401 | 0.4 | 54 | 10 | False |
| market_basket | 0.03 | mlxtend | 5/5 | 0.023 | 0.02 | 0.029 | 8.7 | 54 | 10 | - |
| mushroom | 0.4 | apriori | 30/30 | 1.784 | 1.778 | 2.004 | 4.0 | 565 | 7020 | False |
| mushroom | 0.4 | eclat | 30/30 | 0.718 | 0.714 | 0.777 | 2.5 | 565 | 7020 | False |
| mushroom | 0.4 | fpgrowth | 30/30 | 0.849 | 0.848 | 0.9 | 2.0 | 565 | 7020 | False |
| mushroom | 0.4 | mlxtend | 5/5 | 0.298 | 0.292 | 0.338 | 3.9 | 565 | 7020 | - |
| online_retail | 0.01 | apriori | 5/5 | 496.482 | 494.96 | 539.665 | 48.1 | 1840 | 2108 | False |
| online_retail | 0.01 | eclat | 5/5 | 4.973 | 4.992 | 5.023 | 5.4 | 1840 | 2108 | False |
| online_retail | 0.01 | fpgrowth | 5/5 | 55.792 | 55.891 | 56.518 | 19.5 | 1840 | 2108 | False |
| online_retail | 0.01 | mlxtend | 5/5 | 85.076 | 82.617 | 108.475 | 9650.3 | 1840 | 2108 | - |
| online_retail | 0.02 | apriori | 5/5 | 70.874 | 71.012 | 73.181 | 32.4 | 353 | 151 | False |
| online_retail | 0.02 | eclat | 5/5 | 1.919 | 1.914 | 1.972 | 4.7 | 353 | 151 | False |
| online_retail | 0.02 | fpgrowth | 5/5 | 17.99 | 18.069 | 18.167 | 4.8 | 353 | 151 | False |
| online_retail | 0.02 | mlxtend | 5/5 | 1.986 | 1.978 | 2.115 | 2328.7 | 353 | 151 | - |
| online_retail_ii | 0.01 | apriori | 5/5 | 406.134 | 374.082 | 528.637 | 26.7 | 1216 | 911 | False |
| online_retail_ii | 0.01 | eclat | 5/5 | 3.897 | 3.931 | 3.964 | 6.4 | 1216 | 911 | False |
| online_retail_ii | 0.01 | fpgrowth | 5/5 | 40.369 | 40.717 | 41.075 | 42.7 | 1216 | 911 | False |
| online_retail_ii | 0.01 | mlxtend | 5/5 | 42.825 | 41.69 | 50.23 | 9785.3 | 1216 | 911 | - |
| online_retail_ii | 0.02 | apriori | 5/5 | 64.113 | 64.142 | 64.943 | 26.1 | 282 | 66 | False |
| online_retail_ii | 0.02 | eclat | 5/5 | 1.81 | 1.814 | 1.839 | 6.1 | 282 | 66 | False |
| online_retail_ii | 0.02 | fpgrowth | 5/5 | 15.782 | 15.632 | 16.495 | 3.2 | 282 | 66 | False |
| online_retail_ii | 0.02 | mlxtend | 5/5 | 2.301 | 2.15 | 2.671 | 1950.6 | 282 | 66 | - |
| pumsb | 0.9 | apriori | 5/5 | 332.762 | 323.13 | 374.769 | 87.7 | 2607 | 71474 | False |
| pumsb | 0.9 | eclat | 5/5 | 19.882 | 20.117 | 20.157 | 222.4 | 2607 | 71474 | False |
| pumsb | 0.9 | fpgrowth | 5/5 | 21.337 | 21.233 | 21.738 | 26.0 | 2607 | 71474 | False |
| pumsb | 0.9 | mlxtend | 5/5 | 3.905 | 3.972 | 4.084 | 397.6 | 2607 | 71474 | - |
| pumsb_star | 0.85 | apriori | 5/5 | 10.592 | 10.585 | 10.739 | 85.6 | 0 | 0 | False |
| pumsb_star | 0.85 | eclat | 5/5 | 5.695 | 5.7 | 5.812 | 17.5 | 0 | 0 | False |
| pumsb_star | 0.85 | fpgrowth | 5/5 | 0.044 | 0.046 | 0.054 | 0.3 | 0 | 0 | False |
| pumsb_star | 0.85 | mlxtend | 0/5 | 0 | 0.0 | 0.0 | 0 | - | - | - |
| retail | 0.02 | apriori | 30/30 | 8.443 | 8.417 | 8.651 | 30.1 | 55 | 60 | False |
| retail | 0.02 | eclat | 30/30 | 2.582 | 2.362 | 3.523 | 7.4 | 55 | 60 | False |
| retail | 0.02 | fpgrowth | 30/30 | 7.282 | 7.277 | 7.46 | 0.6 | 55 | 60 | False |
| retail | 0.02 | mlxtend | 5/5 | 31.9 | 32.197 | 36.838 | 2022.8 | 55 | 60 | - |
| retail | 0.03 | apriori | 30/30 | 7.287 | 7.279 | 7.41 | 31.8 | 32 | 28 | False |
| retail | 0.03 | eclat | 30/30 | 2.387 | 2.369 | 2.554 | 9.2 | 32 | 28 | False |
| retail | 0.03 | fpgrowth | 30/30 | 7.206 | 7.225 | 7.58 | 1.1 | 32 | 28 | False |
| retail | 0.03 | mlxtend | 5/5 | 15.003 | 14.947 | 15.338 | 2667.7 | 32 | 28 | - |
| synthetic_dense_5000 | 0.3 | apriori | 5/5 | 13.674 | 13.618 | 14.231 | 4.9 | 1737 | 5262 | False |
| synthetic_dense_5000 | 0.3 | eclat | 5/5 | 2.58 | 2.532 | 2.731 | 16.1 | 1737 | 5262 | False |
| synthetic_dense_5000 | 0.3 | fpgrowth | 5/5 | 55.949 | 57.168 | 58.866 | 16.3 | 1737 | 5262 | False |
| synthetic_dense_5000 | 0.3 | mlxtend | 5/5 | 0.551 | 0.529 | 0.603 | 380.1 | 1737 | 5262 | - |
| synthetic_dense_5000 | 0.5 | apriori | 5/5 | 1.42 | 1.287 | 2.069 | 13.7 | 60 | 20 | False |
| synthetic_dense_5000 | 0.5 | eclat | 5/5 | 0.547 | 0.545 | 0.59 | 9.3 | 60 | 20 | False |
| synthetic_dense_5000 | 0.5 | fpgrowth | 5/5 | 9.9 | 10.43 | 11.636 | 10.6 | 60 | 20 | False |
| synthetic_dense_5000 | 0.5 | mlxtend | 5/5 | 0.025 | 0.026 | 0.027 | 7.4 | 60 | 20 | - |
| synthetic_sparse_2000 | 0.02 | apriori | 5/5 | 0.441 | 0.445 | 0.473 | 0.8 | 269 | 407 | False |
| synthetic_sparse_2000 | 0.02 | eclat | 5/5 | 0.085 | 0.089 | 0.097 | 0.3 | 269 | 407 | False |
| synthetic_sparse_2000 | 0.02 | fpgrowth | 5/5 | 0.325 | 0.277 | 0.544 | 12.7 | 269 | 407 | False |
| synthetic_sparse_2000 | 0.02 | mlxtend | 5/5 | 0.086 | 0.09 | 0.1 | 3.5 | 269 | 407 | - |
| synthetic_sparse_2000 | 0.05 | apriori | 5/5 | 0.19 | 0.195 | 0.202 | 1.0 | 77 | 118 | False |
| synthetic_sparse_2000 | 0.05 | eclat | 5/5 | 0.056 | 0.053 | 0.068 | 0.8 | 77 | 118 | False |
| synthetic_sparse_2000 | 0.05 | fpgrowth | 5/5 | 0.179 | 0.181 | 0.208 | 0.2 | 77 | 118 | False |
| synthetic_sparse_2000 | 0.05 | mlxtend | 5/5 | 0.034 | 0.035 | 0.039 | 0.8 | 77 | 118 | - |
| t10i4d100k | 0.01 | apriori | 1/5 | 592.899 | 592.899 | 592.899 | 0.0 | 385 | 17 | False |
| t10i4d100k | 0.01 | eclat | 5/5 | 6.495 | 6.392 | 7.527 | 49.2 | 385 | 17 | False |
| t10i4d100k | 0.01 | fpgrowth | 5/5 | 72.629 | 71.452 | 77.401 | 292.7 | 385 | 17 | False |
| t10i4d100k | 0.01 | mlxtend | 1/5 | 101.864 | 101.864 | 101.864 | 0.0 | 385 | 17 | - |
| t10i4d100k | 0.02 | apriori | 5/5 | 164.486 | 169.661 | 169.981 | 0.8 | 155 | 0 | False |
| t10i4d100k | 0.02 | eclat | 5/5 | 4.779 | 4.798 | 5.006 | 23.6 | 155 | 0 | False |
| t10i4d100k | 0.02 | fpgrowth | 5/5 | 48.414 | 48.075 | 49.813 | 124.5 | 155 | 0 | False |
| t10i4d100k | 0.02 | mlxtend | 5/5 | 34.395 | 40.949 | 42.953 | 17.3 | 155 | 0 | - |
| tic_tac_toe | 0.3 | apriori | 30/30 | 0.069 | 0.07 | 0.085 | 0.0 | 25 | 10 | False |
| tic_tac_toe | 0.3 | eclat | 30/30 | 0.024 | 0.023 | 0.031 | 0.1 | 25 | 10 | False |
| tic_tac_toe | 0.3 | fpgrowth | 30/30 | 0.102 | 0.1 | 0.138 | 0.3 | 25 | 10 | False |
| tic_tac_toe | 0.3 | mlxtend | 5/5 | 0.012 | 0.013 | 0.015 | 0.0 | 25 | 10 | - |
| tic_tac_toe | 0.5 | apriori | 30/30 | 0.046 | 0.045 | 0.058 | 0.0 | 1 | 0 | False |
| tic_tac_toe | 0.5 | eclat | 30/30 | 0.022 | 0.02 | 0.03 | 0.0 | 1 | 0 | False |
| tic_tac_toe | 0.5 | fpgrowth | 30/30 | 0.045 | 0.044 | 0.064 | 0.0 | 1 | 0 | False |
| tic_tac_toe | 0.5 | mlxtend | 5/5 | 0.005 | 0.005 | 0.006 | 0.0 | 1 | 0 | - |

## Correctness (trial-0 itemset agreement, tol 1e-9)

- accidents@0.5 fpgrowth_vs_eclat: match=True (n_a=8057, n_b=8057, max_support_diff=0.0, reason=None)
- accidents@0.6 apriori_vs_fpgrowth: match=True (n_a=2074, n_b=2074, max_support_diff=0.0, reason=None)
- accidents@0.6 apriori_vs_eclat: match=True (n_a=2074, n_b=2074, max_support_diff=0.0, reason=None)
- accidents@0.6 fpgrowth_vs_eclat: match=True (n_a=2074, n_b=2074, max_support_diff=0.0, reason=None)
- accidents@0.6 skarm_apriori_vs_mlxtend: match=True (n_a=2074, n_b=2074, max_support_diff=0.0, reason=None)
- adult_census@0.05 apriori_vs_fpgrowth: match=True (n_a=11390, n_b=11390, max_support_diff=0.0, reason=None)
- adult_census@0.05 apriori_vs_eclat: match=True (n_a=11390, n_b=11390, max_support_diff=0.0, reason=None)
- adult_census@0.05 fpgrowth_vs_eclat: match=True (n_a=11390, n_b=11390, max_support_diff=0.0, reason=None)
- adult_census@0.05 skarm_apriori_vs_mlxtend: match=True (n_a=11390, n_b=11390, max_support_diff=0.0, reason=None)
- adult_census@0.1 apriori_vs_fpgrowth: match=True (n_a=3216, n_b=3216, max_support_diff=0.0, reason=None)
- adult_census@0.1 apriori_vs_eclat: match=True (n_a=3216, n_b=3216, max_support_diff=0.0, reason=None)
- adult_census@0.1 fpgrowth_vs_eclat: match=True (n_a=3216, n_b=3216, max_support_diff=0.0, reason=None)
- adult_census@0.1 skarm_apriori_vs_mlxtend: match=True (n_a=3216, n_b=3216, max_support_diff=0.0, reason=None)
- balance_scale@0.2 apriori_vs_fpgrowth: match=True (n_a=22, n_b=22, max_support_diff=0.0, reason=None)
- balance_scale@0.2 apriori_vs_eclat: match=True (n_a=22, n_b=22, max_support_diff=0.0, reason=None)
- balance_scale@0.2 fpgrowth_vs_eclat: match=True (n_a=22, n_b=22, max_support_diff=0.0, reason=None)
- balance_scale@0.2 skarm_apriori_vs_mlxtend: match=True (n_a=22, n_b=22, max_support_diff=0.0, reason=None)
- balance_scale@0.3 apriori_vs_fpgrowth: match=True (n_a=2, n_b=2, max_support_diff=0.0, reason=None)
- balance_scale@0.3 apriori_vs_eclat: match=True (n_a=2, n_b=2, max_support_diff=0.0, reason=None)
- balance_scale@0.3 fpgrowth_vs_eclat: match=True (n_a=2, n_b=2, max_support_diff=0.0, reason=None)
- balance_scale@0.3 skarm_apriori_vs_mlxtend: match=True (n_a=2, n_b=2, max_support_diff=0.0, reason=None)
- bms_pos@0.02 apriori_vs_fpgrowth: match=True (n_a=308, n_b=308, max_support_diff=0.0, reason=None)
- bms_pos@0.02 apriori_vs_eclat: match=True (n_a=308, n_b=308, max_support_diff=0.0, reason=None)
- bms_pos@0.02 fpgrowth_vs_eclat: match=True (n_a=308, n_b=308, max_support_diff=0.0, reason=None)
- bms_pos@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=308, n_b=308, max_support_diff=0.0, reason=None)
- bms_webview_1@0.005 apriori_vs_fpgrowth: match=True (n_a=201, n_b=201, max_support_diff=0.0, reason=None)
- bms_webview_1@0.005 apriori_vs_eclat: match=True (n_a=201, n_b=201, max_support_diff=0.0, reason=None)
- bms_webview_1@0.005 fpgrowth_vs_eclat: match=True (n_a=201, n_b=201, max_support_diff=0.0, reason=None)
- bms_webview_1@0.005 skarm_apriori_vs_mlxtend: match=True (n_a=201, n_b=201, max_support_diff=0.0, reason=None)
- bms_webview_1@0.01 apriori_vs_fpgrowth: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- bms_webview_1@0.01 apriori_vs_eclat: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- bms_webview_1@0.01 fpgrowth_vs_eclat: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- bms_webview_1@0.01 skarm_apriori_vs_mlxtend: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- bms_webview_2@0.005 apriori_vs_fpgrowth: match=True (n_a=408, n_b=408, max_support_diff=0.0, reason=None)
- bms_webview_2@0.005 apriori_vs_eclat: match=True (n_a=408, n_b=408, max_support_diff=0.0, reason=None)
- bms_webview_2@0.005 fpgrowth_vs_eclat: match=True (n_a=408, n_b=408, max_support_diff=0.0, reason=None)
- bms_webview_2@0.005 skarm_apriori_vs_mlxtend: match=True (n_a=408, n_b=408, max_support_diff=0.0, reason=None)
- bms_webview_2@0.01 apriori_vs_fpgrowth: match=True (n_a=81, n_b=81, max_support_diff=0.0, reason=None)
- bms_webview_2@0.01 apriori_vs_eclat: match=True (n_a=81, n_b=81, max_support_diff=0.0, reason=None)
- bms_webview_2@0.01 fpgrowth_vs_eclat: match=True (n_a=81, n_b=81, max_support_diff=0.0, reason=None)
- bms_webview_2@0.01 skarm_apriori_vs_mlxtend: match=True (n_a=81, n_b=81, max_support_diff=0.0, reason=None)
- car_evaluation@0.2 apriori_vs_fpgrowth: match=True (n_a=31, n_b=31, max_support_diff=0.0, reason=None)
- car_evaluation@0.2 apriori_vs_eclat: match=True (n_a=31, n_b=31, max_support_diff=0.0, reason=None)
- car_evaluation@0.2 fpgrowth_vs_eclat: match=True (n_a=31, n_b=31, max_support_diff=0.0, reason=None)
- car_evaluation@0.2 skarm_apriori_vs_mlxtend: match=True (n_a=31, n_b=31, max_support_diff=0.0, reason=None)
- car_evaluation@0.3 apriori_vs_fpgrowth: match=True (n_a=12, n_b=12, max_support_diff=0.0, reason=None)
- car_evaluation@0.3 apriori_vs_eclat: match=True (n_a=12, n_b=12, max_support_diff=0.0, reason=None)
- car_evaluation@0.3 fpgrowth_vs_eclat: match=True (n_a=12, n_b=12, max_support_diff=0.0, reason=None)
- car_evaluation@0.3 skarm_apriori_vs_mlxtend: match=True (n_a=12, n_b=12, max_support_diff=0.0, reason=None)
- chess@0.85 apriori_vs_fpgrowth: match=True (n_a=2669, n_b=2669, max_support_diff=0.0, reason=None)
- chess@0.85 apriori_vs_eclat: match=True (n_a=2669, n_b=2669, max_support_diff=0.0, reason=None)
- chess@0.85 fpgrowth_vs_eclat: match=True (n_a=2669, n_b=2669, max_support_diff=0.0, reason=None)
- chess@0.85 skarm_apriori_vs_mlxtend: match=True (n_a=2669, n_b=2669, max_support_diff=0.0, reason=None)
- connect@0.95 apriori_vs_fpgrowth: match=True (n_a=2201, n_b=2201, max_support_diff=0.0, reason=None)
- connect@0.95 apriori_vs_eclat: match=True (n_a=2201, n_b=2201, max_support_diff=0.0, reason=None)
- connect@0.95 fpgrowth_vs_eclat: match=True (n_a=2201, n_b=2201, max_support_diff=0.0, reason=None)
- connect@0.95 skarm_apriori_vs_mlxtend: match=True (n_a=2201, n_b=2201, max_support_diff=0.0, reason=None)
- docword_enron@0.5 apriori_vs_fpgrowth: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- docword_enron@0.5 apriori_vs_eclat: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- docword_enron@0.5 fpgrowth_vs_eclat: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- docword_enron@0.7 apriori_vs_fpgrowth: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- docword_enron@0.7 apriori_vs_eclat: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- docword_enron@0.7 fpgrowth_vs_eclat: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- docword_nips@0.5 apriori_vs_fpgrowth: match=True (n_a=31582, n_b=31582, max_support_diff=0.0, reason=None)
- docword_nips@0.5 apriori_vs_eclat: match=True (n_a=31582, n_b=31582, max_support_diff=0.0, reason=None)
- docword_nips@0.5 fpgrowth_vs_eclat: match=True (n_a=31582, n_b=31582, max_support_diff=0.0, reason=None)
- docword_nips@0.5 skarm_apriori_vs_mlxtend: match=True (n_a=31582, n_b=31582, max_support_diff=0.0, reason=None)
- groceries@0.01 apriori_vs_fpgrowth: match=True (n_a=333, n_b=333, max_support_diff=0.0, reason=None)
- groceries@0.01 apriori_vs_eclat: match=True (n_a=333, n_b=333, max_support_diff=0.0, reason=None)
- groceries@0.01 fpgrowth_vs_eclat: match=True (n_a=333, n_b=333, max_support_diff=0.0, reason=None)
- groceries@0.01 skarm_apriori_vs_mlxtend: match=True (n_a=333, n_b=333, max_support_diff=0.0, reason=None)
- groceries@0.02 apriori_vs_fpgrowth: match=True (n_a=122, n_b=122, max_support_diff=0.0, reason=None)
- groceries@0.02 apriori_vs_eclat: match=True (n_a=122, n_b=122, max_support_diff=0.0, reason=None)
- groceries@0.02 fpgrowth_vs_eclat: match=True (n_a=122, n_b=122, max_support_diff=0.0, reason=None)
- groceries@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=122, n_b=122, max_support_diff=0.0, reason=None)
- kosarak@0.005 fpgrowth_vs_eclat: match=True (n_a=1618, n_b=1618, max_support_diff=0.0, reason=None)
- kosarak@0.01 apriori_vs_fpgrowth: match=True (n_a=383, n_b=383, max_support_diff=0.0, reason=None)
- kosarak@0.01 apriori_vs_eclat: match=True (n_a=383, n_b=383, max_support_diff=0.0, reason=None)
- kosarak@0.01 fpgrowth_vs_eclat: match=True (n_a=383, n_b=383, max_support_diff=0.0, reason=None)
- kosarak_spmf@0.01 apriori_vs_fpgrowth: match=True (n_a=377, n_b=377, max_support_diff=0.0, reason=None)
- kosarak_spmf@0.01 apriori_vs_eclat: match=True (n_a=377, n_b=377, max_support_diff=0.0, reason=None)
- kosarak_spmf@0.01 fpgrowth_vs_eclat: match=True (n_a=377, n_b=377, max_support_diff=0.0, reason=None)
- market_basket@0.02 apriori_vs_fpgrowth: match=True (n_a=103, n_b=103, max_support_diff=0.0, reason=None)
- market_basket@0.02 apriori_vs_eclat: match=True (n_a=103, n_b=103, max_support_diff=0.0, reason=None)
- market_basket@0.02 fpgrowth_vs_eclat: match=True (n_a=103, n_b=103, max_support_diff=0.0, reason=None)
- market_basket@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=103, n_b=103, max_support_diff=0.0, reason=None)
- market_basket@0.03 apriori_vs_fpgrowth: match=True (n_a=54, n_b=54, max_support_diff=0.0, reason=None)
- market_basket@0.03 apriori_vs_eclat: match=True (n_a=54, n_b=54, max_support_diff=0.0, reason=None)
- market_basket@0.03 fpgrowth_vs_eclat: match=True (n_a=54, n_b=54, max_support_diff=0.0, reason=None)
- market_basket@0.03 skarm_apriori_vs_mlxtend: match=True (n_a=54, n_b=54, max_support_diff=0.0, reason=None)
- mushroom@0.4 apriori_vs_fpgrowth: match=True (n_a=565, n_b=565, max_support_diff=0.0, reason=None)
- mushroom@0.4 apriori_vs_eclat: match=True (n_a=565, n_b=565, max_support_diff=0.0, reason=None)
- mushroom@0.4 fpgrowth_vs_eclat: match=True (n_a=565, n_b=565, max_support_diff=0.0, reason=None)
- mushroom@0.4 skarm_apriori_vs_mlxtend: match=True (n_a=565, n_b=565, max_support_diff=0.0, reason=None)
- online_retail@0.01 apriori_vs_fpgrowth: match=True (n_a=1840, n_b=1840, max_support_diff=0.0, reason=None)
- online_retail@0.01 apriori_vs_eclat: match=True (n_a=1840, n_b=1840, max_support_diff=0.0, reason=None)
- online_retail@0.01 fpgrowth_vs_eclat: match=True (n_a=1840, n_b=1840, max_support_diff=0.0, reason=None)
- online_retail@0.01 skarm_apriori_vs_mlxtend: match=True (n_a=1840, n_b=1840, max_support_diff=0.0, reason=None)
- online_retail@0.02 apriori_vs_fpgrowth: match=True (n_a=353, n_b=353, max_support_diff=0.0, reason=None)
- online_retail@0.02 apriori_vs_eclat: match=True (n_a=353, n_b=353, max_support_diff=0.0, reason=None)
- online_retail@0.02 fpgrowth_vs_eclat: match=True (n_a=353, n_b=353, max_support_diff=0.0, reason=None)
- online_retail@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=353, n_b=353, max_support_diff=0.0, reason=None)
- online_retail_ii@0.01 apriori_vs_fpgrowth: match=True (n_a=1216, n_b=1216, max_support_diff=0.0, reason=None)
- online_retail_ii@0.01 apriori_vs_eclat: match=True (n_a=1216, n_b=1216, max_support_diff=0.0, reason=None)
- online_retail_ii@0.01 fpgrowth_vs_eclat: match=True (n_a=1216, n_b=1216, max_support_diff=0.0, reason=None)
- online_retail_ii@0.01 skarm_apriori_vs_mlxtend: match=True (n_a=1216, n_b=1216, max_support_diff=0.0, reason=None)
- online_retail_ii@0.02 apriori_vs_fpgrowth: match=True (n_a=282, n_b=282, max_support_diff=0.0, reason=None)
- online_retail_ii@0.02 apriori_vs_eclat: match=True (n_a=282, n_b=282, max_support_diff=0.0, reason=None)
- online_retail_ii@0.02 fpgrowth_vs_eclat: match=True (n_a=282, n_b=282, max_support_diff=0.0, reason=None)
- online_retail_ii@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=282, n_b=282, max_support_diff=0.0, reason=None)
- pumsb@0.9 apriori_vs_fpgrowth: match=True (n_a=2607, n_b=2607, max_support_diff=0.0, reason=None)
- pumsb@0.9 apriori_vs_eclat: match=True (n_a=2607, n_b=2607, max_support_diff=0.0, reason=None)
- pumsb@0.9 fpgrowth_vs_eclat: match=True (n_a=2607, n_b=2607, max_support_diff=0.0, reason=None)
- pumsb@0.9 skarm_apriori_vs_mlxtend: match=True (n_a=2607, n_b=2607, max_support_diff=0.0, reason=None)
- pumsb_star@0.85 apriori_vs_fpgrowth: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- pumsb_star@0.85 apriori_vs_eclat: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- pumsb_star@0.85 fpgrowth_vs_eclat: match=True (n_a=0, n_b=0, max_support_diff=0.0, reason=None)
- retail@0.02 apriori_vs_fpgrowth: match=True (n_a=55, n_b=55, max_support_diff=0.0, reason=None)
- retail@0.02 apriori_vs_eclat: match=True (n_a=55, n_b=55, max_support_diff=0.0, reason=None)
- retail@0.02 fpgrowth_vs_eclat: match=True (n_a=55, n_b=55, max_support_diff=0.0, reason=None)
- retail@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=55, n_b=55, max_support_diff=0.0, reason=None)
- retail@0.03 apriori_vs_fpgrowth: match=True (n_a=32, n_b=32, max_support_diff=0.0, reason=None)
- retail@0.03 apriori_vs_eclat: match=True (n_a=32, n_b=32, max_support_diff=0.0, reason=None)
- retail@0.03 fpgrowth_vs_eclat: match=True (n_a=32, n_b=32, max_support_diff=0.0, reason=None)
- retail@0.03 skarm_apriori_vs_mlxtend: match=True (n_a=32, n_b=32, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.3 apriori_vs_fpgrowth: match=True (n_a=1737, n_b=1737, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.3 apriori_vs_eclat: match=True (n_a=1737, n_b=1737, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.3 fpgrowth_vs_eclat: match=True (n_a=1737, n_b=1737, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.3 skarm_apriori_vs_mlxtend: match=True (n_a=1737, n_b=1737, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.5 apriori_vs_fpgrowth: match=True (n_a=60, n_b=60, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.5 apriori_vs_eclat: match=True (n_a=60, n_b=60, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.5 fpgrowth_vs_eclat: match=True (n_a=60, n_b=60, max_support_diff=0.0, reason=None)
- synthetic_dense_5000@0.5 skarm_apriori_vs_mlxtend: match=True (n_a=60, n_b=60, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.02 apriori_vs_fpgrowth: match=True (n_a=269, n_b=269, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.02 apriori_vs_eclat: match=True (n_a=269, n_b=269, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.02 fpgrowth_vs_eclat: match=True (n_a=269, n_b=269, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=269, n_b=269, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.05 apriori_vs_fpgrowth: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.05 apriori_vs_eclat: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.05 fpgrowth_vs_eclat: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- synthetic_sparse_2000@0.05 skarm_apriori_vs_mlxtend: match=True (n_a=77, n_b=77, max_support_diff=0.0, reason=None)
- t10i4d100k@0.01 apriori_vs_fpgrowth: match=True (n_a=385, n_b=385, max_support_diff=0.0, reason=None)
- t10i4d100k@0.01 apriori_vs_eclat: match=True (n_a=385, n_b=385, max_support_diff=0.0, reason=None)
- t10i4d100k@0.01 fpgrowth_vs_eclat: match=True (n_a=385, n_b=385, max_support_diff=0.0, reason=None)
- t10i4d100k@0.01 skarm_apriori_vs_mlxtend: match=True (n_a=385, n_b=385, max_support_diff=0.0, reason=None)
- t10i4d100k@0.02 apriori_vs_fpgrowth: match=True (n_a=155, n_b=155, max_support_diff=0.0, reason=None)
- t10i4d100k@0.02 apriori_vs_eclat: match=True (n_a=155, n_b=155, max_support_diff=0.0, reason=None)
- t10i4d100k@0.02 fpgrowth_vs_eclat: match=True (n_a=155, n_b=155, max_support_diff=0.0, reason=None)
- t10i4d100k@0.02 skarm_apriori_vs_mlxtend: match=True (n_a=155, n_b=155, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.3 apriori_vs_fpgrowth: match=True (n_a=25, n_b=25, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.3 apriori_vs_eclat: match=True (n_a=25, n_b=25, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.3 fpgrowth_vs_eclat: match=True (n_a=25, n_b=25, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.3 skarm_apriori_vs_mlxtend: match=True (n_a=25, n_b=25, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.5 apriori_vs_fpgrowth: match=True (n_a=1, n_b=1, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.5 apriori_vs_eclat: match=True (n_a=1, n_b=1, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.5 fpgrowth_vs_eclat: match=True (n_a=1, n_b=1, max_support_diff=0.0, reason=None)
- tic_tac_toe@0.5 skarm_apriori_vs_mlxtend: match=True (n_a=1, n_b=1, max_support_diff=0.0, reason=None)

## Skips, timeouts and errors (recorded, never hidden)

- accidents 0.5 apriori: trial 0 exceeded 600 s; remaining trials skipped
- kosarak 0.01 mlxtend: dense boolean DataFrame would need ~40.9 GB
- kosarak 0.005 apriori: trial 0 exceeded 600 s; remaining trials skipped
- kosarak 0.005 mlxtend: dense boolean DataFrame would need ~40.9 GB
- kosarak_spmf 0.01 mlxtend: dense boolean DataFrame would need ~5.3 GB

## Threshold rationale

- accidents: Large dense traffic data (340,183 transactions). Probes: at 0.50 FP-Growth OK (119.6 s) and Eclat OK (278.5 s), Apriori timed out, mlxtend was killed. 0.60 is added as a faster second point. Loader must exclude the leading '340183 468' header line.
- adult_census: Medium census set (32,561 transactions, 6.19% density); 0.10 primary, 0.05 lower-support comparison.
- balance_scale: Very small categorical set (625 transactions); two cheap points.
- bms_pos: Large point-of-sale click-stream (515,596 transactions). Probes already showed 0.01 failing (Apriori timeout twice, FP-Growth killed twice), so only 0.02 is attempted.
- bms_webview_1: Small click-stream with very short transactions (avg length 2.51); low thresholds are safe. File is tab-separated; the loader sniffs the delimiter.
- bms_webview_2: Medium click-stream (77,512 transactions, avg length 4.62); low thresholds safe.
- car_evaluation: Very small categorical set (1,728 transactions); two cheap points.
- chess: Very dense (49.3%) with long transactions; 0.85 is the previously verified point. Lower thresholds explode rule counts (0.80 already yields ~552k rules), so a single high threshold is used.
- connect: Very dense (33.3%) Connect-4 positions, 67,557 transactions. Probes already showed 0.90 failing (Apriori timeout, FP-Growth process killed), so only 0.95 is attempted; if it also fails it is recorded as a skip with the probe evidence.
- docword_enron: Large document-word set (39,861 documents, avg length 93.1). Higher thresholds than NIPS because the set is 26x larger; 0.70 primary, 0.50 second point. If 0.50 times out it is recorded as a skip.
- docword_nips: Document-word set (1,500 documents, avg length 497.5). Probes: at 0.50 all four engines OK and agreeing exactly (31,582 itemsets, 1,148,468 rules); at 0.30 all four timed out. Only 0.50 is attempted.
- groceries: Classic sparse retail; 0.02 is the paper's verified point (122 itemsets, 49 rules, all engines agreeing); 0.01 adds a lower-support comparison.
- kosarak: Largest set (990,002 click-stream transactions) but very sparse (0.02%). Low thresholds are feasible because few items are frequent; 0.01 is the primary point, 0.005 the lower-support comparison.
- kosarak_spmf: Smaller SPMF sequence variant of kosarak (187,140 sequences). SPMF -1/-2 tokens must be stripped by the loader. Single threshold.
- market_basket: Small sparse retail; both thresholds previously verified with all engines.
- mushroom: Dense (19.3%) with 23-item transactions; 0.40 is the previously verified point. Single threshold to bound rule explosion on dense data.
- online_retail: Retail with long transactions (avg length 25.1); 0.02 primary, 0.01 lower-support comparison. Cancelled invoices and non-positive quantities were already dropped in conversion.
- online_retail_ii: Second retail set, same shape as online_retail (avg length 22.6); same thresholds for comparability.
- pumsb: Dense census data with very long transactions (avg length 74). Only a high threshold is attempted to avoid candidate explosion; lower thresholds are not attempted, not hidden.
- pumsb_star: Sparser census variant (avg length 50.5). Single high threshold for the same density reason as pumsb.
- retail: Large sparse retail; 0.03 is the verified point for all three engines. 0.02 is included for a lower-support comparison; Apriori may time out there and that outcome is recorded, not hidden.
- synthetic_dense_5000: Locally generated dense set (seed 42, 61.1% density, only 50 items); small and fast, two points.
- synthetic_sparse_2000: Locally generated sparse set (seed 42, planted itemset {1,2,3}); small and fast, two points.
- t10i4d100k: Standard synthetic benchmark (100,000 transactions); 0.01-0.02 is the published comparison range for this set.
- tic_tac_toe: Very small categorical set (958 transactions); any threshold runs in seconds, so two points are cheap.

## Trial-count rule (read before citing)

30-trial statistics (p50/p95/mean) are claimed ONLY for the subset: groceries, chess, mushroom, retail, connect, bms_pos, tic_tac_toe, docword_nips. All other datasets report 5 trials; mlxtend always reports 5 trials where feasible.

