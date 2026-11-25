# skarm: Scalable Association-Rule Mining Estimator with Apriori Baseline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn Compatible](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)
[![Tests: 264](https://img.shields.io/badge/tests-264%20passing-brightgreen)](#)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

> **KL University Capstone Project Framework: KLCAP-2026-00340**  
> **Faculty Guide / Supervisor**: Dr. Dudigam Ramya  
> **Specialization**: AI for Computational Intelligence  
> **Related Specializations**: Data Engineering for AI · SOFTWARE MODELLING & DEVOPS  
> **Upstream Context & Positioning**: Inspired by `scikit-learn/scikit-learn#32712` ("Add Apriori Algorithm for Association Rule Mining"), which was **closed as `not_planned`** on 2025-11-16 by maintainer `@virchan` (citing `#2662`, placing frequent-pattern mining outside core scikit-learn scope). Accordingly, `skarm` is positioned strictly as an **independent, third-party scikit-learn-compatible library**—not an accepted or endorsed scikit-learn core feature.

---

## Overview

`skarm` is an independent, auditable, scikit-learn-compatible Python library for frequent-itemset mining and association-rule discovery with bounded candidate growth, certified numerical metrics, peak-memory telemetry, and safe degradation modes.

It unifies:
- **Baseline Engine**: Conventional Apriori algorithm with candidate generation and support counting.
- **Advanced Contributions**: FP-Growth (Grahne & Zhu FP-Array co-occurrence pruning, bitmask triangle/depth-3 verification, bit-packed transaction compression, and prefix-tree mining) and Eclat (vertical C-level arbitrary-precision bitset TID intersections via `np.packbits` and `int.bit_count()`, with exact weighted/ultra-large sparse fallback).
- **Safety & Pruning Engine**: Dynamic candidate explosion ceiling (`max_candidates`), maximum length control (`max_len`), process memory governor (`max_memory_mb`), and bounded degradation designed to safeguard computational resources against exponential candidate explosion on dense transactional datasets.
- **Scikit-Learn-Compatible Estimator**: Conforms to `BaseEstimator` and `TransformerMixin` conventions (`fit`, `predict`, `transform`, `clone`, `joblib` serialization) as a standalone third-party package.
- **Rule Metrics**: Support, confidence, lift, leverage, conviction, and Zhang's metric verified against analytical hand-computed oracles to machine precision ($< 10^{-15}$).

---

## Repository Structure

```text
capstone/
├── skarm/                           # Main package source code
│   ├── __init__.py                  # Package exports
│   ├── preprocessing.py             # TransactionEncoder (Dense & CSR sparse)
│   ├── apriori.py                   # Conventional Apriori baseline miner
│   ├── fpgrowth.py                  # FP-Growth prefix-tree pattern miner
│   ├── eclat.py                     # Eclat vertical TID-list miner
│   ├── rules.py                     # Association rule generator & metric formulas
│   ├── pruning.py                   # Candidate bounding & degraded mode policy
│   ├── estimator.py                 # Scikit-learn AssociationRulesEstimator
│   ├── telemetry.py                 # TelemetryLogger & RecoveryCoordinator
│   ├── explorer.py                  # Interactive RuleExplorer & markdown tables
│   └── cli.py                       # Command-line interface
├── tests/                           # Verification and acceptance test suite
│   ├── __init__.py
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── oracle_data.py           # Hand-computed ground truth oracle
│   ├── test_encoder.py              # Unit tests for TransactionEncoder
│   ├── test_apriori.py              # Apriori baseline verification
│   ├── test_fpgrowth.py             # FP-Growth verification
│   ├── test_eclat.py                # Eclat vertical miner verification
│   ├── test_rules.py                # Rule metrics & oracle accuracy tests
│   ├── test_estimator.py            # Scikit-learn API compliance tests
│   ├── test_check_estimator_compatibility.py  # Official sklearn check_estimator audit
│   ├── test_acceptance_contract.py  # Strict KPI-4 acceptance contract test
│   ├── test_challenger_fallback.py  # Runner fallback & provenance regression tests
│   ├── test_negative_cases.py       # Mandatory negative tests (NT-1 through NT-6)
│   ├── e2e/                         # 5-tier independent E2E suite (88 tests)
│   └── run_acceptance.py            # Automated Acceptance Harness (AC-1..AC-4)
├── benchmarks/                      # Performance & resource profiling
│   ├── run_benchmarks.py            # Scalability runner across N=200, 1000, 3000
│   ├── run_pc_campaign.py           # 25-dataset campaign runner
│   ├── run_baselines.py             # SPMF & mlxtend baseline comparison runner
│   ├── run_bounded_ablation.py      # Candidate & memory governor ablation
│   └── benchmark_results.json       # Versioned benchmark execution results
├── paper/                           # Research manuscripts, slides & figures
│   ├── paper-ieee.tex / .pdf        # IEEEtran manuscript
│   ├── paper-ieee-graphical.pdf     # Graphical abstract edition
│   ├── paper.tex / .pdf             # Full-length research manuscript
│   ├── presentation_slides.md       # Viva defense slides
│   └── figures/                     # Architecture, engine & ablation plots
├── public/                          # Interactive Web Explorer frontend
├── server.mjs                       # Node.js server & Python web bridge
├── scripts/                         # Build, LaTeX compile & verification scripts
├── docs/                            # Capstone Deliverables (D1 through D7 & reports)
│   ├── D1_PROBLEM_CHARTER_AND_SPECIFICATION.md
│   ├── D2_CP1_APRIORI_BASELINE_EVALUATION.md
│   ├── D3_CP2_CONTRIBUTION_AND_TRADEOFFS.md
│   ├── D4_SYSTEM_INTEGRATION_AND_OPERABILITY.md
│   ├── D5_ACCEPTANCE_TEST_REPORT_AND_KPIS.md
│   ├── D6_REPRODUCIBILITY_RUNBOOK.md
│   ├── D7_EXECUTIVE_DEFENSE_AND_PORTFOLIO.md
│   └── SKLEARN_COMPATIBILITY_REPORT.md
├── CITATION.cff                     # Machine-readable citation metadata
├── DATASET_INVENTORY.md             # 25-dataset manifest, licenses & SHA-256 hashes
├── HANDOFF.md                       # Executive handoff & reproduction summary
├── pyproject.toml                   # Build system and project metadata
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Master project documentation
```

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Basic Example
```python
from skarm.estimator import AssociationRulesEstimator

# Transactions dataset
transactions = [
    ["milk", "bread", "butter"],
    ["bread", "butter"],
    ["milk", "bread"],
    ["beer", "diapers"],
    ["milk", "beer", "diapers", "bread"],
    ["bread", "butter", "milk"],
    ["diapers", "beer"],
    ["milk", "bread", "butter", "eggs"],
]

# Instantiate and fit estimator
est = AssociationRulesEstimator(
    min_support=0.3,
    metric="confidence",
    min_threshold=0.6,
    algorithm="auto",  # Automatically selects fpgrowth or eclat based on density
)
est.fit(transactions)

# Print discovered rules
rules = est.get_rules()
print(rules[["antecedent_names", "consequent_names", "support", "confidence", "lift"]])

# Generate recommendations for a customer basket
recommendations = est.predict([["beer"]])
print("Recommendation for ['beer']:", recommendations)
# Output: [['diapers']]
```

### 3. Command Line Interface on Real-World Datasets
Run on canonical production market basket data:
```bash
# Groceries market basket analysis (9,835 transactions):
python -m skarm.cli --dataset groceries --min-support 0.02 --min-confidence 0.25 --top-n 10

# Retail dataset high-throughput mining (88,162 transactions):
python -m skarm.cli --dataset retail --min-support 0.02 --algorithm eclat --top-n 10
```

### 4. Interactive Web Explorer
Launch the Node.js web interface backed by `scripts/web_bridge.py` for interactive dataset mining, rule inspection, and IEEE manuscript preview:
```bash
npm install
npm start
```

---

## Verification & Multi-Tier Test Suite
 
Run the comprehensive test suite (264 automated tests: 176 unit/integration/challenge/compatibility/contract + 88 5-tier E2E):
```bash
pytest -v
```

Execute the 5-Tier End-to-End (E2E) test runner:
```bash
python tests/e2e/run_e2e.py
```
*(88 tests across Feature Coverage, Boundaries, Pairwise Interactions, Real-World Scenarios, and Adversarial Hardening)*

Execute the independent acceptance test harness (AC-1..AC-4 and NT-1..NT-6, strict 30-trial warmup KPI-4 with zero small-runtime exemptions):
```bash
python tests/run_acceptance.py
```
*(Demonstrates representative precision, negative failure boundaries, held-out evaluation mechanics on a 70/30 synthetic split, and strict p95 runtime comparison)*

Run empirical benchmark campaigns across synthetic scales (5 warmup + 30 measured trials per engine, zero exemptions):
```bash
python benchmarks/run_benchmarks.py
```

---

## Key Performance Indicator (KPI) Scorecard

| ID | Measure | Baseline (O2 Apriori) | Achieved (CP2 FP-Growth & Eclat) | Target Rule | Verdict |
|---|---|---|---|---|---|
| **KPI-1** | Frequent-itemset precision & recall vs oracle (%) | 100.0% / 100.0% | **100.0% / 100.0%** ($\Delta_{\mathrm{supp}} = 0.0$) | $\ge 98.0\%$ | **PASS** |
| **KPI-2** | Rule-metric numerical error rubric | Rubric 80 | **100.0/100** ($2.22 \times 10^{-16}$) | Score $\ge 80/100$ | **PASS** |
| **KPI-3** | Peak candidate count | 105 candidates | **0 – 46 candidates** (capped on dense stress) | $\ge 5\%$ reduction / capped | **PASS** |
| **KPI-4** | Runtime by transaction scale (strict 30-trial p95, no exemptions) | Reference p95 | **AC-4**: FP `0.501×`, Eclat `0.153×`; **Scales 200–3000**: FP `0.157–0.267×` (engine) / `0.243–0.315×` (est), Eclat `0.086–0.374×` (engine) / `0.266–0.343×` (est); 0 failures | $p95 \le 0.80 \times$ Baseline | **PASS** |
| **KPI-5** | Peak memory (`tracemalloc`) & bounded memory envelope | 0.027 – 0.129 MB (synthetic engine) | **0.028 – 0.180 MB** (synthetic scales: fixed buffer floor `1.00–1.39×`; candidate/memory governor prevents OOM on dense explosion) | Bounded / monitored | **MONITORED / BOUNDED** |
| **KPI-6** | Estimator API compliance pass rate | 100.0% | **100.0% (7/7 capstone API contracts)**; generic `check_estimator`: 14 passed, 39 domain xfails, 1 skipped, 0 unexpected ([report](docs/SKLEARN_COMPATIBILITY_REPORT.md)) | 100.0% | **PASS** |

---

## Deliverables Index
- [D1: Project Charter & System Architecture](docs/D1_PROBLEM_CHARTER_AND_SPECIFICATION.md)
- [D2: CP1 Apriori Baseline Evaluation](docs/D2_CP1_APRIORI_BASELINE_EVALUATION.md)
- [D3: CP2 Contributions & Trade-offs](docs/D3_CP2_CONTRIBUTION_AND_TRADEOFFS.md)
- [D4: System Integration & Operability](docs/D4_SYSTEM_INTEGRATION_AND_OPERABILITY.md)
- [D5: Independent Acceptance & NT-1..6 Audit](docs/D5_ACCEPTANCE_TEST_REPORT_AND_KPIS.md)
- [D6: Reproducibility Runbook](docs/D6_REPRODUCIBILITY_RUNBOOK.md)
- [D7: Viva Defense & Portfolio Dossier](docs/D7_EXECUTIVE_DEFENSE_AND_PORTFOLIO.md)
- [Scikit-Learn Compatibility Report](docs/SKLEARN_COMPATIBILITY_REPORT.md)
- [Dataset Inventory & Provenance](DATASET_INVENTORY.md)
- [Executive Handoff](HANDOFF.md)
- [Citation Metadata (CFF)](CITATION.cff)
