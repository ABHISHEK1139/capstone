# SKARM Test Infrastructure & Multi-Tier Verification Framework

**Project**: SKARM (Scalable Association-Rule Mining Estimator with Apriori Baseline)  
**Capstone Reference**: KLCAP-2026-00340 (KL University)  
**Publication Target**: IEEE TKDE / ACM TKDD Camera-Ready Standards  
**Maintainer**: Independent E2E Test Engineering Track  

---

## 1. Overview & Architecture

The SKARM testing infrastructure provides an independent, multi-layered verification harness designed to enforce strict scikit-learn API interoperability, algorithmic equivalence across mining engines (Apriori, FP-Growth, Eclat), mathematical precision against analytical oracles, and resource bounds under extreme data densities.

The test infrastructure is structured into three complementary testing layers:
1. **Unit & Negative Tests (`tests/test_*.py`)**: Component-level regression testing, boundary parameter validation, and negative recovery testing (NT-1 through NT-6).
2. **Automated Acceptance Harness (`tests/run_acceptance.py`)**: Formal validation of acceptance criteria (AC-1 through AC-4) and KPIs (KPI-1 itemset recall/precision, KPI-2 numerical error rubric, KPI-3 candidate bounding, KPI-4 speedup).
3. **Independent End-to-End (E2E) Verification Suite (`tests/e2e/`)**: Requirement-driven, opaque-box multi-tier test campaign exercising public interfaces as an end user across all supported data formats, estimators, pipelines, and industry scenarios.

```
tests/
├── fixtures/
│   └── oracle_data.py               # Analytical oracle frequent itemsets and rules
├── e2e/                              # Independent E2E Multi-Tier Test Suite
│   ├── __init__.py
│   ├── test_tier1_features.py       # Tier 1: Feature Coverage (32 tests)
│   ├── test_tier2_boundaries.py     # Tier 2: Boundary & Corner Cases (29 tests)
│   ├── test_tier3_pairwise.py       # Tier 3: Cross-Feature Interactions (6 tests)
│   ├── test_tier4_scenarios.py      # Tier 4: Real-World Industry Scenarios (5 tests)
│   ├── test_tier5_adversarial.py    # Tier 5: Adversarial Coverage Hardening (16 tests)
│   ├── run_e2e.py                   # Self-contained programmatic runner
│   └── e2e_results.json             # Structured JSON test report
├── test_apriori.py                  # Unit tests for Apriori BFS baseline
├── test_fpgrowth.py                 # Unit tests for FP-Growth prefix-tree
├── test_eclat.py                    # Unit tests for Eclat vertical TID-list
├── test_encoder.py                  # Unit tests for TransactionEncoder
├── test_estimator.py                # Unit tests for AssociationRulesEstimator
├── test_negative_cases.py           # Negative test campaigns NT-1..NT-6
├── test_rules.py                    # Unit tests for rule metrics & Zhang's formula
└── run_acceptance.py                # Acceptance harness (AC-1..AC-4 & KPI scorecard)
```

---

## 2. Multi-Tier E2E Test Architecture

The E2E test suite adheres to a 5-tier progressive verification model (including Tier 5 adversarial coverage hardening):

### Tier 1: Feature Coverage (`test_tier1_features.py` — 32 Tests)
Verifies basic contracts and isolated features against analytical ground truth (>= 5 tests per feature):
- **Feature 1.1: Input Format Handling (6 tests)**: Raw transaction list-of-lists, dense NumPy boolean arrays, Scipy CSR sparse matrices, Scipy CSC sparse matrices, Pandas DataFrames with preserved column headers, and `TransactionEncoder` dense/sparse modes.
- **Feature 1.2: Core Mining Algorithms (6 tests)**: Apriori level-wise BFS, FP-Growth pattern growth, and Eclat vertical intersection matching hand-computed analytical oracles (`ORACLE_FREQUENT_ITEMSETS_0_4`); cross-engine support equivalence ($\Delta\mathrm{supp} < 10^{-9}$); automatic engine selection; `max_len` length restrictions.
- **Feature 1.3: Rule Generation and Metrics (6 tests)**: Support, confidence, and lift accuracy against analytical ground truth; infinite conviction on deterministic rules ($conf = 1.0$); zero leverage at statistical independence; Zhang's metric strictly bounded in $[-1.0, 1.0]$; multi-metric filtering; non-empty disjoint antecedent/consequent partition constraints.
- **Feature 1.4: Execution Result States (6 tests)**: Tri-state execution model (`EXACT`, `DEGRADED`, `FAILED`); `is_complete_` and `is_degraded_` flags; candidate explosion exception raising; candidate truncation policy; candidate count semantics (`unvisited_candidates_status`); diagnostic `termination_reason_` reporting.
- **Feature 1.5: Estimator API Contracts (8 tests)**: Parameter immutability and `sklearn.base.clone()`; standard fitted attributes (`rules_`, `columns_`, `n_features_in_`, `feature_names_in_`); recommendation generation via `predict()`; binary rule firing indicators via `transform()`; sparse matrix output support; feature names tracking via `get_feature_names_out()`; model evaluation via `.score()`; `joblib` serialization round-trip.

### Tier 2: Boundary & Corner Cases (`test_tier2_boundaries.py` — 29 Tests)
Tests extreme parameters, unusual input geometries, missing values, and resource budgets:
- **Feature 2.1: Transaction Size & Content Boundaries (6 tests)**: Empty datasets `[]` raising `ValueError`; datasets composed of empty transactions `[[], []]`; single-item transactions generating singletons and 0 rules; 100% density all-ones matrices; single transaction databases ($n=1$); 100 disjoint singletons.
- **Feature 2.2: Duplicate & Token Idempotency (5 tests)**: Identical duplicate transactions contributing proportional support $k/n$; repeated tokens in single transactions (`['A', 'A', 'B']`) treated idempotently; token permutation invariance (`['A', 'B']` vs `['B', 'A']`); case-sensitive distinct tokens (`'Milk'` vs `'milk'`); whitespace, dashes, commas, and special characters.
- **Feature 2.3: Extreme Thresholds (6 tests)**: `min_support=1.0` (universal items only); `min_support=0.0` (all observed combinations); `min_threshold=1.0` (deterministic rules only); `min_threshold=0.0` (all non-trivial partitions); out-of-bounds threshold validation; unknown metric name rejection.
- **Feature 2.4: Missing Value Tokens & Validation (6 tests)**: Filtering of `None` tokens; filtering of `np.nan` and `float('nan')` tokens; rejection of continuous float matrices with `ValueError`; rejection of integer count matrices ($>1$); rejection of negative matrix values; rejection of 1D arrays.
- **Feature 2.5: Candidate & Resource Budgets (6 tests)**: `max_candidates=1` with `degrade` entering safe mode; `max_candidates=1` with `raise` throwing `CandidateExplosionError`; `max_candidates=1` with `truncate` capping candidates; non-positive `max_candidates` rejection; non-positive `max_len` rejection; `max_len=1` singleton discovery.

### Tier 3: Cross-Feature Combinations (`test_tier3_pairwise.py` — 6 Tests)
Validates pairwise feature interactions across data formats, algorithms, and pipelines:
- **Test 3.1**: Sparse CSR matrix + Eclat + candidate budget degradation + metadata inspection (`future_candidates_status == "UNKNOWN"`).
- **Test 3.2**: Dense DataFrame + FP-Growth + `transform()` rule indicators + downstream `LogisticRegression` classification pipeline.
- **Test 3.3**: Transaction list + Apriori + `GridSearchCV` hyperparameter tuning with `mean_confidence_scorer` from `skarm.metrics`.
- **Test 3.4**: Sample weights + multi-engine equivalence check (weighted duplicates vs explicit `sample_weight` across Apriori, FP-Growth, and Eclat).
- **Test 3.5**: Sparse CSR + auto algorithm selection + Zhang's metric filtering + `predict()` recommendation inference.
- **Test 3.6**: Multi-dimensional resource budgets (candidates + runtime) + cross-engine status consistency.

### Tier 4: Real-World Application Scenarios (`test_tier4_scenarios.py` — 5 Tests)
Validates end-to-end user workflows in realistic industry operational contexts:
- **Scenario 4.1: Retail Market Basket Analysis (Groceries Workflow)**: Complete grocery receipt workflow from raw text items through encoding, mining, rule filtering by lift $> 1.2$, and generating product recommendations for incomplete shopper baskets.
- **Scenario 4.2: E-Commerce Recommendation Engine in Scikit-Learn Pipeline**: Full classification pipeline combining `AssociationRulesEstimator` transformer, `VarianceThreshold` feature selection, and `LogisticRegression` classifier predicting user conversion.
- **Scenario 4.3: Clickstream Session Pattern Mining with High Sparsity (<0.1% Density)**: 2,000 web browsing sessions across 500 URLs with density $< 0.1\%$ evaluated via vertical Eclat without memory explosion, discovering an embedded conversion funnel.
- **Scenario 4.4: Dense Survey / Census Demographic Pattern Discovery**: 150 respondents with 25 questions (density $\sim 40\%$) with `max_len=3` and candidate bounding preventing combinatorial explosion while discovering demographic correlations.
- **Scenario 4.5: Fault-Tolerant Batch ETL Pipeline with Burst Recovery**: Continuous streaming ETL processing normal batches in `EXACT` mode, catching a dense burst anomaly into `DEGRADED` mode, and recovering immediately to `EXACT` mode on normal traffic without state leakage.

### Tier 5: Adversarial Coverage Hardening (`test_tier5_adversarial.py` — 16 Tests)
White-box adversarial stress testing against boundary gaps, resource budgets, persistence, and pipeline APIs:
- **Feature 5.1: Extreme Boundary Matrices (3 tests)**: All-zeros dense and CSR sparse matrices producing 0 itemsets and exact status; single-cell (1x1), single-column (Nx1), and single-transaction (1xN) matrices; 100% density combinatorial stress with cross-engine oracle equivalence ($\Delta\mathrm{supp} = 0$).
- **Feature 5.2: Multi-Dimensional Budget Transitions (3 tests)**: Rapid candidate degradation and subsequent clean recovery on re-fit without state leakage; time budget (`max_time_seconds`) enforcement under degrade and raise policies; memory budget (`max_memory_mb`) governor and constructor boundary parameter validation.
- **Feature 5.3: Model Persistence & Deserialization (2 tests)**: Python `pickle` and `joblib` streaming round-trips for both `EXACT` and `DEGRADED` models, verifying full preservation of `status_`, `is_complete_`, `itemsets_`, `rules_`, `metadata_`, `predict()`, `transform()`, and `score()`.
- **Feature 5.4: Metric Numerical Stability & Mathematical Invariants (2 tests)**: Ultra-low support regime ($10^{-5}$) on 100,000 transactions across Apriori, FP-Growth, and Eclat without underflow; mathematical limit invariants for leverage ($[-0.25, 0.25]$), Zhang's metric ($[-1.0, 1.0]$), conviction ($\infty$ for deterministic rules), and zero NaNs.
- **Feature 5.5: Scikit-Learn Pipeline & ColumnTransformer with pandas Output (3 tests)**: `Pipeline.set_output(transform="pandas")` yielding formatted rule columns; `ColumnTransformer.set_output(transform="pandas")` combining rule features and passthrough columns; zero-rules pandas handling.
- **Feature 5.6: Adversarial Distributions & Query Fuzzing (3 tests)**: Heavy outlier transaction sample weights ($10^6$ weight dominance) and sample weight error checking; malformed, novel, and empty query fuzzing; `sklearn.base.clone()` parameter immutability and unfitted clone verification.

---

## 3. Requirement Traceability Matrix

| Requirement / Gap ID | Description | Primary Test Location | Verification Method |
|:---|:---|:---|:---|
| **R1, Gap 3** | Evaluated 3-tier determinism (result, ordering, execution) | `tests/test_negative_cases.py` (NT-5), `tests/e2e/test_tier1_features.py` | Hash stability, rule order equality |
| **R2, Gap 6** | Formal mathematical problem specification ($D, I, T_i, supp, conf$) | `tests/e2e/test_tier1_features.py` | Analytical oracle comparison |
| **R2, Gap 7** | Rigorous data semantics (duplicates, idempotency, missing tokens) | `tests/e2e/test_tier2_boundaries.py` | Token deduplication, NaN/None filtering |
| **R2, Gap 8** | Valid rule partitions ($X, Y \neq \emptyset, X \cap Y = \emptyset, X \cup Y \in \mathcal{F}_\sigma$) | `tests/e2e/test_tier1_features.py` | Set intersection and union checks |
| **R2, Gap 9** | Tri-state execution model (`EXACT`, `DEGRADED`, `FAILED`) | `tests/e2e/test_tier1_features.py`, `tests/e2e/test_tier4_scenarios.py` | `status_`, `is_complete_`, `is_degraded_` |
| **R2, Gap 10** | Candidate count semantics (exact unvisited vs unknown) | `tests/e2e/test_tier1_features.py`, `tests/e2e/test_tier3_pairwise.py` | `metadata_['future_candidates_status']` |
| **R3, Gap 11** | Scikit-learn estimator conventions (`clone()`, fitted attributes) | `tests/e2e/test_tier1_features.py` | `sklearn.base.clone()`, `feature_names_in_` |
| **R3, Gap 12** | Pipeline interoperability (`fit`, `predict`, `transform`, `get_feature_names_out`) | `tests/e2e/test_tier1_features.py`, `tests/e2e/test_tier3_pairwise.py`, `tests/e2e/test_tier4_scenarios.py` | `Pipeline([AssociationRulesEstimator, Clf])` |
| **R3, Gap 13** | Model selection scoring functions & `.score()` | `tests/e2e/test_tier1_features.py`, `tests/e2e/test_tier3_pairwise.py` | `GridSearchCV`, `skarm.metrics` |
| **R3, Gap 14** | Algorithm operation counters (joins, trees, intersections) | `tests/e2e/test_tier3_pairwise.py`, `tests/run_acceptance.py` | `metadata_['counters']` |
| **R3, Gap 15** | Multi-dimensional resource budgets (candidates, time, memory) | `tests/e2e/test_tier2_boundaries.py`, `tests/e2e/test_tier3_pairwise.py` | `max_candidates`, `max_time_seconds` |
| **R3, Gap 16** | Safety governor harmonization across Apriori, FP, Eclat | `tests/e2e/test_tier3_pairwise.py`, `tests/run_acceptance.py` | Multi-engine degradation consistency |
| **R4, Gap 21** | Bug remediation (B1 Zhang's metric, B2 NaN tokens, B4 DataFrame columns, B8 names out, B9 score) | `tests/e2e/test_tier1_features.py`, `tests/e2e/test_tier2_boundaries.py` | Direct boundary and API assertions |

---

## 4. Authoritative Expected Output Derivation

For every test case, expected outputs are grounded in explicit mathematical definitions and analytical oracles:
1. **Hand-Computed Analytical Oracle (`tests/fixtures/oracle_data.py`)**:
   - `HAND_COMPUTED_TRANSACTIONS`: 5-transaction multiset with items in $\{A, B, C, D, E, F\}$.
   - `ORACLE_FREQUENT_ITEMSETS_0_4`: Exactly 9 frequent itemsets derived by manual counting:
     $$\text{supp}(A) = 4/5 = 0.8, \quad \text{supp}(B) = \text{supp}(C) = 3/5 = 0.6, \quad \text{supp}(E) = 2/5 = 0.4$$
     $$\text{supp}(AB) = 2/5 = 0.4, \quad \text{supp}(AC) = 3/5 = 0.6, \quad \text{supp}(BC) = 2/5 = 0.4, \quad \text{supp}(BE) = 2/5 = 0.4$$
     $$\text{supp}(ABC) = 2/5 = 0.4$$
   - `ORACLE_RULES_0_4_CONF_0_6`: Exactly 11 association rules with pre-computed analytical values for support, confidence, lift, leverage, and conviction.
2. **Mathematical Invariant Bounds**:
   - $\text{supp}(X \to Y) \in [0.0, 1.0]$
   - $\text{conf}(X \to Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)} \in [0.0, 1.0]$
   - $\text{lift}(X \to Y) = \frac{\text{conf}(X \to Y)}{\text{supp}(Y)} \in [0.0, \infty)$
   - $\text{lev}(X \to Y) = \text{supp}(X \cup Y) - \text{supp}(X)\,\text{supp}(Y) \in [-0.25, 0.25]$
   - $\text{conv}(X \to Y) = \frac{1 - \text{supp}(Y)}{1 - \text{conf}(X \to Y)} \in [0.0, \infty]$
   - $\text{zhang}(X \to Y) \in [-1.0, 1.0]$
3. **Set-Theoretic Properties**:
   - Token idempotency: $|\{A, A, B\}| = |\{A, B\}| = 2$.
   - Universal item property: at $\sigma = 1.0$, only $X \subseteq \bigcap_{i=1}^n T_i$ are returned.

---

## 5. Execution Instructions

### Running the E2E Verification Suite via Standalone Runner
The self-contained runner executes all 5 tiers, collects fine-grained telemetry, and outputs a formatted terminal scorecard and JSON report:
```bash
python tests/e2e/run_e2e.py
```

### Running via Pytest
```bash
# Run all E2E tiers with verbose output
python -m pytest tests/e2e/ -v

# Run a specific tier
python -m pytest tests/e2e/test_tier1_features.py -v
python -m pytest tests/e2e/test_tier2_boundaries.py -v
python -m pytest tests/e2e/test_tier3_pairwise.py -v
python -m pytest tests/e2e/test_tier4_scenarios.py -v
python -m pytest tests/e2e/test_tier5_adversarial.py -v
```

### Running the Acceptance Test Harness
```bash
python tests/run_acceptance.py
```

### Running the Complete Repository Test Campaign
```bash
python -m pytest tests/
```

---

## 6. Verification Summary & Test Scorecard

| Tier | Category | Test File | Test Count | Pass Count | Status | Execution Time |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Tier 1** | Feature Coverage | `test_tier1_features.py` | 32 | 32 | **PASS** | 2.06 s |
| **Tier 2** | Boundary & Corner Cases | `test_tier2_boundaries.py` | 29 | 29 | **PASS** | 0.31 s |
| **Tier 3** | Cross-Feature Combinations | `test_tier3_pairwise.py` | 6 | 6 | **PASS** | 0.61 s |
| **Tier 4** | Real-World Application Scenarios | `test_tier4_scenarios.py` | 5 | 5 | **PASS** | 0.63 s |
| **Tier 5** | Adversarial Coverage Hardening | `test_tier5_adversarial.py` | 16 | 16 | **PASS** | 4.82 s |
| **TOTAL** | **All 5 Tiers** | **`tests/e2e/`** | **88** | **88** | **ALL PASSED** | **8.43 s** |
