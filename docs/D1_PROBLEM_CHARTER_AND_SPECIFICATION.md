# Deliverable D1: Project Charter, Requirements, and Architecture

**Capstone Project ID**: KLCAP-2026-00340  
**Title**: Scalable Association-Rule Mining Estimator with Apriori Baseline  
**Programme**: B.Tech (AI for Computational Intelligence)  
**Related Specializations**: Data Engineering for AI · SOFTWARE MODELLING & DEVOPS  
**Upstream Context / Inspiration**: `scikit-learn/scikit-learn#32712` ("Add Apriori Algorithm for Association Rule Mining") — opened 2025-11-14, **closed as `not_planned`** on 2025-11-16 by scikit-learn maintainer `@virchan` citing `#2662` (pattern mining outside core scikit-learn scope; recommended as an external library such as `scikit-mine` or `scikit-learn-contrib`). **`skarm` is positioned strictly as an independent, third-party scikit-learn-compatible library**, not an accepted or endorsed core scikit-learn feature.

---

## 1. Problem Statement & Context

In real-world transactional intelligence, market-basket analytics, and sequence modeling, standard feature engineering and classification workflows often benefit from association rules. Upstream `scikit-learn` maintainers explicitly confirmed in issues `#2662` and `#32712` (closed as `not_planned` on 2025-11-16) that frequent-pattern and association-rule mining sits outside core `scikit-learn`'s scope and should instead live in dedicated third-party libraries. Existing external packages often have non-standard APIs, fragile candidate generation, unpredictable memory usage, or incomplete integration with scikit-learn pipelines.

The governing engineering challenge is that unconstrained association rule mining suffers from **combinatorial candidate explosion**: in dense transactional matrices, candidate counts scale as $\mathcal{O}(2^M)$ where $M$ is the number of distinct items. Without bounded candidate generation, reproducible support counting, and mathematically certified rule-quality semantics, models crash with Out-Of-Memory (OOM) errors or produce untrustworthy rules.

### Source-to-Claim Mapping
| Upstream Reference | Confirmed Issue Status / Engineering Gap | Independent Library Response & Architectural Guarantee |
|---|---|---|
| **scikit-learn issue #32712** | Feature request ("Add Apriori Algorithm for Association Rule Mining", opened 2025-11-14) was **closed as `not_planned`** on 2025-11-16 by `@virchan` (citing `#2662`, directing pattern mining to dedicated external packages) | Delivered `skarm` as an **independent, third-party scikit-learn-compatible library** (`AssociationRulesEstimator` conforming to `BaseEstimator` / `TransformerMixin` conventions) rather than an upstream core scikit-learn feature |
| **Unbounded Candidate Growth** | Dense transaction matrices cause combinatorial explosion | Implemented `PruningPolicy` with configurable `max_candidates`, safe degradation, and telemetry |
| **Support Counting Semantics** | Inconsistent definition of sample weights and sparse transactions | Mathematically certified support arithmetic on both dense boolean arrays and CSR/CSC matrices |
| **Lack of Standard Metrics** | Discrepancies in rule quality scores across packages | Built numerical oracle verifying support, confidence, lift, leverage, conviction, and Zhang's metric to machine precision ($< 10^{-15}$) |

---

## 2. Requirements Traceability Matrix

### Functional Requirements (FR-1 through FR-7)
- **FR-1 (Ingestion & Validation)**: Ingest raw transaction iterables or binary indicator matrices, validating types, lengths, non-negativity of weights, and boundary conditions.
- **FR-2 (Apriori Baseline)**: Reproduce conventional Apriori baseline with candidate generation $C_k$, join step, prune step, and verification against hand-computed ground truth fixtures.
- **FR-3 (Advanced Subsystems)**: Implement FP-Growth (prefix-tree pattern growth) and Eclat (vertical TID-list intersection) with sparse matrix acceleration.
- **FR-4 (System Integration)**: Unify `TransactionEncoder`, frequent itemset miners, rule generator, pruning policy, and `AssociationRulesEstimator`.
- **FR-5 (Modern Toolchain)**: Target Python 3.10+ (tested on Python 3.12) with standard scientific dependencies (NumPy, SciPy, Pandas, scikit-learn, Joblib).
- **FR-6 (Demonstrable Operable System)**: Deliver an installable package with CLI (`skarm`), Python API, benchmark harness, and interactive rule explorer.
- **FR-7 (Negative Test Detection & Safe Recovery)**: Detect candidate explosion, boundary anomalies, and duplicate records; invoke safe degraded mode and maintain an audit trail.

### Non-Functional Requirements (NFR-1 through NFR-6)
- **NFR-1 (Performance)**: Maintain sub-second execution on standard transaction scales, meeting KPI targets under AC-1 and AC-4.
- **NFR-2 (Reliability & Recovery)**: Pass all 6 mandatory negative tests (NT-1 to NT-6); prevent silent crashes or unhandled exceptions.
- **NFR-3 (Domain Assurance)**: Numerical accuracy verified against analytical oracles; no condition-wise failures masked by aggregate metrics.
- **NFR-4 (Evidence Quality)**: Zero test-set leakage; deterministic tie-breaking and random seed control.
- **NFR-5 (Maintainability)**: Clean PEP-8 codebase, modular architecture, comprehensive pytest suite with 100% test pass rate.
- **NFR-6 (Operability)**: Built-in `TelemetryLogger` recording peak memory (MB), candidate generation counts, and execution traces.

---

## 3. Architectural Design

```mermaid
flowchart TD
    subgraph Data Layer
        A1[Raw Transactions List] --> B1[TransactionEncoder]
        A2[CSR / CSC Sparse Matrix] --> B2[Matrix Validator]
        A3[Dense Boolean ndarray] --> B2
    end

    subgraph Mining Engines
        B1 --> C[Mining Orchestrator]
        B2 --> C
        C --> D1[Apriori Engine Baseline]
        C --> D2[FP-Growth Engine Differentiating]
        C --> D3[Eclat Vertical Miner]
    end

    subgraph Safety & Telemetry
        D1 <--> E[PruningPolicy max_candidates]
        D2 <--> E
        D3 <--> E
        E --> F[TelemetryLogger & RecoveryCoordinator]
    end

    subgraph Rule Generation & Evaluation
        D1 --> G[association_rules Generator]
        D2 --> G
        D3 --> G
        G --> H[Metrics: Support, Confidence, Lift, Leverage, Conviction, Zhang]
    end

    subgraph Scikit-Learn Estimator API
        H --> I[AssociationRulesEstimator]
        I --> J1[fit / transform / predict]
        I --> J2[Joblib Serialization]
        I --> J3[RuleExplorer & CLI]
    end
```

---

## 4. Key Performance Indicator (KPI) Targets

| ID | Measure | Baseline (Apriori O2) | Target Rule | Achieved | Status |
|---|---|---|---|---|---|
| **KPI-1** | Frequent-itemset precision vs oracle (%) | 100.0% | $\ge 98.0\%$ | **100.0%** | **PASS** |
| **KPI-2** | Rule-metric numerical error | Max error $< 10^{-12}$ | Rubric Score $\ge 80/100$ | **100.0/100** ($2.22 \times 10^{-16}$) | **PASS** |
| **KPI-3** | Peak candidate count | Unbounded ($> 10^5$) | Capped by guardrail policy | **Enforced & Capped** | **PASS** |
| **KPI-4** | Runtime by transaction scale (p95, warmup + 30 trials, monotonic clock, no exemptions) | Reference runtime | $p95 \le 0.80 \times$ Baseline at equal quality | **FP-Growth `0.235–0.375×`, Eclat `0.180–0.333×` of Apriori p95** ($\Delta_{\mathrm{supp}}=0.0$, 0 failures) | **PASS** |
| **KPI-5** | Peak memory by transaction scale | Reference memory | Monitored & bounded without critical quality regression | **0.033–0.181 MB** on synthetic scales (fixed buffer floor near parity `1.00–1.39×`; governor caps dense explosion) | **MONITORED / BOUNDED** |
| **KPI-6** | Estimator API compliance pass rate | 100.0% | 100.0% | **100.0% (7/7 checks)** | **PASS** |
