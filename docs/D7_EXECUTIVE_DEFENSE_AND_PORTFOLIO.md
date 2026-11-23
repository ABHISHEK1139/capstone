# Deliverable D7: Capstone Defense Portfolio & Viva Presentation Dossier

**Capstone Project ID**: KLCAP-2026-00340  
**Title**: Scalable Association-Rule Mining Estimator with Apriori Baseline  
**Department**: Computer Science & Engineering / Artificial Intelligence  
**Institution**: Koneru Lakshmaiah Education Foundation (KL University)  
**Dominant Specialization**: AI for Computational Intelligence  
**Related Specializations**: Data Engineering for AI · SOFTWARE MODELLING & DEVOPS  

---

## 1. Executive Summary

This culminating engineering capstone project was motivated by **scikit-learn GitHub issue #32712** ("Add Apriori Algorithm for Association Rule Mining", opened 2025-11-14). Upstream `scikit-learn` maintainers **closed #32712 as `not_planned`** on 2025-11-16 (citing `#2662`), confirming that frequent-pattern mining lies outside core `scikit-learn` scope and belongs in a dedicated external library. Accordingly, `skarm` is positioned strictly as an **independent, third-party scikit-learn-compatible library**—not an accepted or endorsed core scikit-learn feature. In industrial data mining and developer tooling, association rule mining is plagued by **candidate combinatorial explosion**, unstandardized APIs, lack of safe memory bounding, and inconsistent support arithmetic across sparse and weighted transactions.

Over the two-semester lifecycle (CP1 and CP2), our team engineered `skarm`, an independent open-source, scikit-learn-compatible library that establishes:
1. **CP1 Reference Baseline (O1, O2)**: A mathematically certified conventional Apriori baseline validated against an analytical hand-computed oracle ($100\%$ precision/recall, error $\le 2.22 \times 10^{-16}$).
2. **CP2 Differentiating Contributions (O3)**: High-performance FP-Growth (FP-Array co-occurrence pruning, bitmask triangle/depth-3 verification, bit-packed compression, and prefix-tree mining) and Eclat (vertical C-level bitset TID intersections) engines, achieving strict $p95 \le 0.80\times$ Apriori runtime across all acceptance and scale workloads (`0.180–0.375×`, zero small-runtime exemptions) with bit-exact support identity ($\Delta_{\mathrm{supp}} = 0.0$).
3. **Safety & Pruning Engine (O4)**: Real-time candidate ceiling guards (`max_candidates`), maximum length limits (`max_len`), on-demand process RSS limits (`max_memory_mb`), and proactive degradation protocols that safeguard system memory under adversarial, high-density transaction conditions.
4. **Independent Acceptance & Robustness Campaign (O5)**: Exhaustive verification across 6 mandatory negative test suites (NT-1 to NT-6), satisfying all Acceptance Conditions (AC-1 to AC-4, including strict 30-trial warmup KPI-4) and achieving a **100% test pass rate across 264 automated tests (176 unit, integration, challenge, compatibility, and contract tests + 88 5-tier E2E tests)**.

---

## 2. Key Contributions & Innovation Highlights

```text
+-----------------------------------------------------------------------------------------+
|                                    skarm Architecture                                   |
+-----------------------------------------------------------------------------------------+
|  scikit-learn API:  AssociationRulesEstimator (BaseEstimator, fit / predict / transform)|
+-----------------------------------------------------------------------------------------+
|  Mining Engines:    Apriori (Baseline) | FP-Growth (Tree) | Eclat (Vertical TID)        |
+-----------------------------------------------------------------------------------------+
|  Safety Governor:   PruningPolicy (Candidate Bound, Degraded Mode, TelemetryLogger)   |
+-----------------------------------------------------------------------------------------+
|  Metrics Engine:    Support, Confidence, Lift, Leverage, Conviction, Zhang's Metric     |
+-----------------------------------------------------------------------------------------+
|  Data Layer:        TransactionEncoder (Dense ndarray & CSR Sparse Matrix)              |
+-----------------------------------------------------------------------------------------+
```

### Quantified KPI Achievements
- **Frequent-Itemset Precision & Recall vs Oracle**: **100.0% / 100.0%** (Target: $\ge 98.0\%$, bit-exact $\Delta_{\mathrm{supp}} = 0.0$).
- **Rule-Metric Numerical Error**: **100.0/100 Rubric Score** (Maximum observed numerical error: $2.22 \times 10^{-16}$).
- **Candidate Bounding**: **56.2%–100.0% reduction** in explicit candidate generations across synthetic scales; hard-capped under dense stress (NT-1, NT-6).
- **Strict 30-Trial Runtime Speedup (KPI-4, Zero Exemptions)**: AC-4 ($1000 \times 12$) FP-Growth `0.315×` (`3.17×` faster) and Eclat `0.180×` (`5.56×` faster) of Apriori p95; synthetic scales ($N=200, 1000, 3000$) FP-Growth `0.235–0.375×` and Eclat `0.213–0.333×` of Apriori p95, 0 failures.
- **Estimator API Compliance**: **100.0% (7/7)** (Full `get_params`, `set_params`, `clone`, `joblib` pickling compatibility as an independent third-party estimator).
- **Negative Test Campaign**: **6/6 passed** (NT-1 to NT-6 covering candidate explosion, token deduplication, sample weights, boundary thresholds, determinism, and stress workloads).

---

## 3. Team Role Allocation & Viva Defense Guide

As mandated by the Capstone Framework, each domain area was spearheaded by dedicated leadership:

### 1. Data & Problem-Evidence Leadership
- **Responsibilities**: Problem charter validation, upstream issue #32712 reproduction, dataset acquisition, schema validation, and `TransactionEncoder` development.
- **Viva Key Topic**: Handling sparse CSR matrix encoding, token deduplication idempotency (NT-2), and sample weight validation (NT-3).

### 2. Model & Algorithm Leadership
- **Responsibilities**: Implementation of the level-wise Apriori baseline, FP-Tree prefix structures, conditional pattern base recursion, and Eclat vertical TID-list intersection.
- **Viva Key Topic**: Algorithmic complexity differences: why FP-Growth requires only 2 database passes vs Apriori's $k$ passes; why Eclat outperforms on sparse matrices.

### 3. Serving & Integration Leadership
- **Responsibilities**: Scikit-learn estimator API integration (`AssociationRulesEstimator`), `fit`/`transform`/`predict` contract, joblib serialization, CLI tooling (`skarm.cli`), and `RuleExplorer`.
- **Viva Key Topic**: How `AssociationRulesEstimator` conforms to scikit-learn conventions (`get_params`, `clone()`, state pickling, no mutable defaults).

### 4. Evaluation & Responsible-AI Leadership
- **Responsibilities**: Design of hand-computed oracle fixtures, KPI-1..6 measurement harnesses, negative test campaigns (NT-1..6), safety guardrail engineering (`PruningPolicy`), and telemetry profiling.
- **Viva Key Topic**: Proactive degradation protocols under combinatorial explosion; mitigating unmonitored memory exhaustion and preventing silent data omissions.

---

## 4. Academic & Industry Impact

By delivering a verified, independent third-party scikit-learn-compatible association rule estimator—aligned with the scikit-learn maintainers' closure of `#2662` and `#32712` directing pattern mining to dedicated external packages—this capstone provides researchers and practitioners with an auditable, pipeline-ready library for retail, healthcare, cybersecurity, and bioinformatics workflows.
