# Deliverable D3: CP2 Advanced Contributions, Comparative Analysis, and Trade-offs

**Capstone Project ID**: KLCAP-2026-00340  
**Phase**: Capstone Project II (CP2 / Objective O3)  
**Deliverable**: D3 — Differentiating Contributions, Pruning Policies, and Trade-off Analysis  

---

## 1. Differentiating Contributions Overview

While the Apriori baseline is theoretically elegant, it generates explosive candidate counts ($C_k$) in dense transactional workloads. To overcome this fundamental bottleneck, CP2 delivers three core engineering contributions:

1. **FP-Growth Miner ([`skarm.fpgrowth`](../skarm/fpgrowth.py))**: Eliminates explicit level-wise candidate generation ($C_k = \emptyset$) by combining Grahne & Zhu's FP-Array pairwise co-occurrence matrix, lossless bitmask triangle and conditional FP-Array depth-3 verification, bit-packed transaction compression, and frequency-ordered FP-Tree conditional pattern-base mining.
2. **Eclat Vertical Miner ([`skarm.eclat`](../skarm/eclat.py))**: Employs vertical transaction identification (TID) bitsets packed via `np.packbits` into C-level arbitrary-precision Python integer masks intersected with hardware-accelerated bitwise `&` and `.bit_count()` (with an exact Python `set` fallback for weighted or ultra-large sparse matrices).
3. **Safety & Pruning Policy Engine ([`skarm.pruning`](../skarm/pruning.py))**: Implements real-time candidate bounding (`max_candidates`), maximum length control (`max_len`), on-demand RSS memory governance (`max_memory_mb`), and safe degradation modes (NT-1, NT-6) to protect against memory exhaustion without imposing polling overhead when memory budgets are disabled.

---

## 2. Theoretical and Architectural Comparison

```text
Algorithm      Candidate Generation Strategy   Data Representation             Scans Required
---------      -----------------------------   -------------------             --------------
Apriori        Level-wise C_k generation       Horizontal binary matrix        k passes
FP-Growth      Zero C_k generation             FP-Array + Prefix Tree          2 passes
Eclat          Equivalence class expansion     Vertical Bitset / TID Lists     1 pass to invert
```

---

## 3. Comparative Benchmark Results Across Workloads

Benchmarked across synthetic transaction scales (`M=15`, 25% item density, `min_support=0.08`, seed 42) under the strict Charter KPI-4 protocol (**5 warmup trials + 30 measured trials**, monotonic `time.perf_counter()`, **zero small-runtime exemptions**, bit-exact cross-engine support identity $\Delta_{\mathrm{supp}} = 0.0$, 0 failures; see `benchmarks/benchmark_results.json`):

### Candidate Generation Bounding (KPI-3)
| Workload | Apriori Peak Candidates | FP-Growth Peak Candidates | Eclat Peak Candidates | Candidate Reduction |
|---|---|---|---|---|
| Scale $N=200$ (39 itemsets) | 105 | 46 | 46 | **56.2% reduction** |
| Scale $N=1,000$ (15 itemsets) | 105 | 0 | 0 | **100.0% reduction** |
| Scale $N=3,000$ (15 itemsets) | 105 | 0 | 0 | **100.0% reduction** |
| Dense Stress ($80\%$ density) | $> 500,000$ (OOM risk) | Tree-bounded | Intersect-bounded | **Capped by Guardrail** |

### Strict 30-Trial Execution Runtime & Peak Memory Profile (KPI-4, KPI-5)
| Scale ($N$) | Algorithm | Core Engine p50 / p95 / p99 (ms) | Engine p95 Ratio vs Apriori | Estimator `fit()` p50 / p95 / p99 (ms) | Est p95 Ratio vs Apriori | Engine Peak Mem (MB) | Exact Match ($\Delta_{\mathrm{supp}}$) |
|---|---|---|---|---|---|---|---|
| **$N = 200$** | Apriori (O2 Ref) | 1.961 / 2.279 / 2.372 ms | 1.000× | 28.341 / 32.956 / 35.017 ms | 1.000× | 0.033 MB | Reference |
| | FP-Growth (CP2) | 0.569 / **0.744** / 0.783 ms | **0.327× (3.06× faster)** | 9.115 / **9.655** / 9.818 ms | **0.293× (3.41× faster)** | 0.034 MB | **0.0** |
| | Eclat (CP2) | 0.249 / **0.723** / 1.552 ms | **0.317× (3.15× faster)** | 6.814 / **7.299** / 7.449 ms | **0.221× (4.51× faster)** | 0.033 MB | **0.0** |
| **$N = 1,000$** | Apriori (O2 Ref) | 1.697 / 2.011 / 2.314 ms | 1.000× | 23.680 / 27.203 / 27.543 ms | 1.000× | 0.078 MB | Reference |
| | FP-Growth (CP2) | 0.386 / **0.527** / 0.597 ms | **0.262× (3.82× faster)** | 5.257 / **6.387** / 7.712 ms | **0.235× (4.26× faster)** | 0.089 MB | **0.0** |
| | Eclat (CP2) | 0.278 / **0.432** / 0.529 ms | **0.215× (4.65× faster)** | 5.228 / **5.800** / 5.985 ms | **0.213× (4.69× faster)** | 0.079 MB | **0.0** |
| **$N = 3,000$** | Apriori (O2 Ref) | 2.099 / 2.400 / 2.464 ms | 1.000× | 24.812 / 33.835 / 35.143 ms | 1.000× | 0.129 MB | Reference |
| | FP-Growth (CP2) | 0.665 / **0.784** / 0.830 ms | **0.327× (3.06× faster)** | 6.305 / **12.692** / 19.218 ms | **0.375× (2.67× faster)** | 0.180 MB | **0.0** |
| | Eclat (CP2) | 0.419 / **0.619** / 0.703 ms | **0.258× (3.88× faster)** | 5.761 / **11.263** / 18.528 ms | **0.333× (3.00× faster)** | 0.129 MB | **0.0** |

*Note on KPI-5 memory*: On these small synthetic matrices ($N \le 3{,}000$, $M=15$), absolute `tracemalloc` peak memory is dominated by fixed NumPy input/boolean matrix buffers (0.03–0.18 MB), so Eclat stays at parity (`1.00–1.01×`) and FP-Growth adds a small temporary chunked `float32` GEMM buffer (`1.01–1.39×`). Memory advantage over Apriori materializes primarily on high-cardinality and dense workloads where Apriori's explicit level-$k$ candidate sets grow combinatorially.

---

## 4. Trade-off Analysis & Recommendation Matrix

| Operational Criterion | Apriori Baseline | FP-Growth Contribution | Eclat Vertical Contribution |
|---|---|---|---|
| **Sparse Data ($< 10\%$ ones)** | Good | Moderate (tree overhead) | **Superior (small TID sets)** |
| **Dense Data ($> 60\%$ ones)** | Fails (candidate explosion) | **Superior (tree compression)** | Moderate (large TID sets) |
| **Memory Footprint** | Low for small items | Moderate (node links) | Moderate to High |
| **Simplicity & Auditability** | High | Medium | High |
| **Automatic Selection** | Baseline benchmark | Chosen for dense inputs (density $\ge 0.15$) | Chosen for sparse inputs (sparsity $\ge 85\%$, density $< 0.15$) |

In `AssociationRulesEstimator(algorithm='auto')`, the engine calculates exact matrix density:
$$\text{Density} = \frac{\text{nnz}}{N \times M}, \quad \text{Sparsity} = 1.0 - \text{Density}$$
- If sparsity $\ge 85\%$ (density $\le 0.15$), it routes to **Eclat** (fast vertical TID-list intersections over compact transaction lists), regardless of whether stored as a sparse SciPy matrix or dense NumPy array.
- Otherwise (density $> 0.15$), it routes to **FP-Growth** for prefix-tree compression.
- Apriori is preserved for verifiable baseline replication.
