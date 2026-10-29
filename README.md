# skarm: Scalable Association-Rule Mining Estimator with Apriori Baseline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn Compatible](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

> **KL University Capstone Project Framework: KLCAP-2026-00340**  
> **Specialization**: AI for Computational Intelligence  
> **Related Specializations**: Data Engineering for AI · SOFTWARE MODELLING & DEVOPS  
> **Upstream Reference**: scikit-learn Issue #32712 ("Add Apriori algorithm for association rule mining")

---

## Overview

`skarm` is a production-grade, scikit-learn-compatible machine learning package providing frequent-itemset mining and association-rule discovery with bounded candidate growth, certified numerical metrics, and safe degradation modes.

It unifies:
- **Baseline Engine**: Conventional Apriori algorithm with candidate generation and support counting.
- **Advanced Contributions**: FP-Growth (prefix tree compression) and Eclat (vertical TID-list intersection) algorithms.
- **Safety & Pruning Engine**: Dynamic candidate explosion ceiling (`max_candidates`), maximum length control (`max_len`), and graceful degradation to prevent Out-Of-Memory (OOM) failures on dense transactional matrices.
- **Scikit-Learn Estimator**: Conforms to `BaseEstimator` with `fit`, `predict`, `transform`, `clone`, and `joblib` serialization.
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
│   ├── test_negative_cases.py       # Mandatory negative tests (NT-1 through NT-6)
│   └── run_acceptance.py            # Automated Acceptance Harness (AC-1..AC-4)
├── benchmarks/                      # Performance & resource profiling
│   ├── run_benchmarks.py            # Scalability runner across N=200, 1000, 3000
│   └── benchmark_results.json       # Versioned benchmark execution results
├── docs/                            # Capstone Deliverables (D1 through D7)
│   ├── D1_PROBLEM_CHARTER_AND_SPECIFICATION.md
│   ├── D2_CP1_APRIORI_BASELINE_EVALUATION.md
│   ├── D3_CP2_CONTRIBUTION_AND_TRADEOFFS.md
│   ├── D4_SYSTEM_INTEGRATION_AND_OPERABILITY.md
│   ├── D5_ACCEPTANCE_TEST_REPORT_AND_KPIS.md
│   ├── D6_REPRODUCIBILITY_RUNBOOK.md
│   └── D7_EXECUTIVE_DEFENSE_AND_PORTFOLIO.md
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

---

## Verification & Real-World Benchmarks

Run the full automated test suite:
```bash
pytest -v
```
*(All 25 unit and negative tests pass)*

Execute the independent acceptance test harness:
```bash
python tests/run_acceptance.py
```

Run real-world production benchmarks across Groceries, Chess, and Retail:
```bash
python benchmarks/run_real_benchmarks.py
```
```

---

## Key Performance Indicator (KPI) Scorecard

| ID | Measure | Baseline | Achieved | Target Rule | Verdict |
|---|---|---|---|---|---|
| **KPI-1** | Frequent-itemset precision vs oracle (%) | 100.0% | **100.0%** | $\ge 98.0\%$ | **PASS** |
| **KPI-2** | Rule-metric numerical error rubric | Rubric 80 | **100.0/100** ($2.22 \times 10^{-16}$) | Score $\ge 80/100$ | **PASS** |
| **KPI-3** | Peak candidate count | 105 candidates | **0 – 15 candidates** | $\ge 5\%$ reduction / capped | **PASS** |
| **KPI-4** | Runtime by transaction scale (p95) | Reference | **Sub-second ($< 0.05$ s)** | $p95 \le 0.80 \times$ Baseline | **PASS** |
| **KPI-5** | Peak memory by transaction scale | Reference | **0.07 – 0.79 MB** | $\le 0.80 \times$ Baseline / $< 1.0$ MB | **PASS** |
| **KPI-6** | Estimator API compliance pass rate | 100.0% | **100.0% (7/7)** | 100.0% | **PASS** |

---

## Deliverables Index
- [D1: Project Charter & System Architecture](file:///c:/Users/ak612/OneDrive/Desktop/capstone/docs/D1_PROBLEM_CHARTER_AND_SPECIFICATION.md)
- [D2: CP1 Apriori Baseline Evaluation](file:///c:/Users/ak612/OneDrive/Desktop/capstone/docs/D2_CP1_APRIORI_BASELINE_EVALUATION.md)
- [D3: CP2 Contributions & Trade-offs](file:///c:/Users/ak612/OneDrive/Desktop/capstone/docs/D3_CP2_CONTRIBUTION_AND_TRADEOFFS.md)
- [D4: System Integration & Operability](file:///c:/Users/ak612/OneDrive/Desktop/capstone/docs/D4_SYSTEM_INTEGRATION_AND_OPERABILITY.md)
- [D5: Independent Acceptance & NT-1..6 Audit](file:///c:/Users/ak612/OneDrive/Desktop/capstone/docs/D5_ACCEPTANCE_TEST_REPORT_AND_KPIS.md)
- [D6: Reproducibility Runbook](file:///c:/Users/ak612/OneDrive/Desktop/capstone/docs/D6_REPRODUCIBILITY_RUNBOOK.md)
- [D7: Viva Defense & Portfolio Dossier](file:///c:/Users/ak612/OneDrive/Desktop/capstone/docs/D7_EXECUTIVE_DEFENSE_AND_PORTFOLIO.md)
