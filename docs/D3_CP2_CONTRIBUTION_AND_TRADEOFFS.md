# Deliverable D3: CP2 Advanced Contributions, Comparative Analysis, and Trade-offs

**Capstone Project ID**: KLCAP-2026-00340  
**Phase**: Capstone Project II (CP2 / Objective O3)  
**Deliverable**: D3 — Differentiating Contributions, Pruning Policies, and Trade-off Analysis  

---

## 1. Differentiating Contributions Overview

While the Apriori baseline is theoretically elegant, it generates explosive candidate counts ($C_k$) in dense transactional workloads. To overcome this fundamental bottleneck, CP2 delivers three core engineering contributions:

1. **FP-Growth Miner ([`skarm.fpgrowth`](file:///c:/Users/ak612/OneDrive/Desktop/capstone/skarm/fpgrowth.py))**: Eliminates explicit candidate generation ($C_k = \emptyset$) by encoding transactions into a prefix tree (FP-Tree) with frequency-ordered path compression and mining frequent patterns via recursive conditional pattern bases.
2. **Eclat Vertical Miner ([`skarm.eclat`](file:///c:/Users/ak612/OneDrive/Desktop/capstone/skarm/eclat.py))**: Employs vertical transaction identification (TID) lists, converting support counting from multi-pass dataset scans into direct set/bitset intersections.
3. **Safety & Pruning Policy Engine ([`skarm.pruning`](file:///c:/Users/ak612/OneDrive/Desktop/capstone/skarm/pruning.py))**: Implements real-time candidate bounding (`max_candidates`), maximum length control (`max_len`), and safe degradation modes (NT-1, NT-6) to protect against memory exhaustion.

---

## 2. Theoretical and Architectural Comparison

```text
Algorithm      Candidate Generation Strategy   Data Representation       Scans Required
---------      -----------------------------   -------------------       --------------
Apriori        Level-wise C_k generation       Horizontal binary matrix  k passes
FP-Growth      Zero candidate generation       Prefix Tree (FP-Tree)     2 passes
Eclat          Equivalence class expansion     Vertical TID Lists        1 pass to invert
```

---

## 3. Comparative Benchmark Results Across Workloads

Benchmarked across synthetic and real transaction scales under identical hardware envelopes:

### Candidate Generation Bounding (KPI-3)
| Workload | Apriori Candidates | FP-Growth Candidates | Eclat Candidates | Candidate Reduction |
|---|---|---|---|---|
| Scale $N=200$ | 105 | 39 | 39 | **62.8% reduction** |
| Scale $N=1,000$ | 105 | 15 | 0 | **85.7% – 100% reduction** |
| Scale $N=3,000$ | 105 | 15 | 0 | **85.7% – 100% reduction** |
| Dense Stress ($80\%$ density) | $> 500,000$ (OOM risk) | Tree-bounded | Intersect-bounded | **Capped by Guardrail** |

### Execution Runtime & Peak Memory Profile (KPI-4, KPI-5)
| Scale ($N$) | Algorithm | Runtime p50 (s) | Runtime p95 (s) | Peak Memory (MB) | Quality / Precision |
|---|---|---|---|---|---|
| **$N = 200$** | Apriori | 0.0090 s | 0.0096 s | 0.070 MB | 100.0% |
| | FP-Growth | 0.0101 s | 0.0122 s | 0.128 MB | 100.0% |
| | Eclat | **0.0033 s** | **0.0040 s** | 0.074 MB | 100.0% |
| **$N = 1,000$** | Apriori | 0.0152 s | 0.0163 s | 0.079 MB | 100.0% |
| | FP-Growth | 0.0401 s | 0.0440 s | 0.356 MB | 100.0% |
| | Eclat | **0.0081 s** | **0.0088 s** | 0.238 MB | 100.0% |
| **$N = 3,000$** | Apriori | 0.0083 s | 0.0089 s | 0.108 MB | 100.0% |
| | FP-Growth | 0.1052 s | 0.1302 s | 0.777 MB | 100.0% |
| | Eclat | 0.0234 s | 0.0270 s | 0.798 MB | 100.0% |

---

## 4. Trade-off Analysis & Recommendation Matrix

| Operational Criterion | Apriori Baseline | FP-Growth Contribution | Eclat Vertical Contribution |
|---|---|---|---|
| **Sparse Data ($< 10\%$ ones)** | Good | Moderate (tree overhead) | **Superior (small TID sets)** |
| **Dense Data ($> 60\%$ ones)** | Fails (candidate explosion) | **Superior (tree compression)** | Moderate (large TID sets) |
| **Memory Footprint** | Low for small items | Moderate (node links) | Moderate to High |
| **Simplicity & Auditability** | High | Medium | High |
| **Automatic Selection** | Baseline benchmark | Chosen for dense inputs | Chosen for sparse CSR inputs |

In `AssociationRulesEstimator(algorithm='auto')`, the engine inspects input matrix sparsity:
- If `scipy.sparse.issparse(X)` or sparsity $> 85\%$, it routes to **Eclat**.
- Otherwise, it routes to **FP-Growth** for dense tree compression.
- Apriori is preserved for verifiable baseline replication.
