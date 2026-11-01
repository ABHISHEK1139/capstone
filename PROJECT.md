# Project: SKARM Camera-Ready Revision (KLCAP-2026-00340)

## Architecture
SKARM is a scikit-learn-style association rule mining framework providing unified estimator and transformer interfaces over three distinct mining engines (Apriori, FP-Growth, Eclat).
- **Core Engine Layer (`skarm/`)**:
  - `preprocessing.py`: `TransactionEncoder` converting transactional iterables to sparse CSR/CSC boolean matrices.
  - `apriori.py`: Breadth-first level-wise candidate generator ($L_{k-1} \bowtie L_{k-1} \to C_k$).
  - `fpgrowth.py`: Depth-first recursive pattern growth with $O(1)$ tail-pointer header table.
  - `eclat.py`: Vertical equivalence class clustering with TID-list intersections.
  - `rules.py`: Association rule generation from frequent itemsets with statistical metrics.
  - `pruning.py`: Multi-dimensional resource governor ($B_{\text{cand}}, B_{\text{mem}}, B_{\text{time}}$).
  - `estimator.py`: Scikit-learn compatible `AssociationRulesEstimator` (final estimator & intermediate transformer).
  - `metrics.py`: Unsupervised scoring metrics (`mean_confidence`, `mean_lift`, `total_support`) for `GridSearchCV`.
- **Benchmarking Layer (`benchmarks/`, `data/`)**:
  - 25 verified benchmark datasets across sparse, dense, long, and short transaction databases.
  - Automated campaign runner, 3-stage timing protocol ($t_{\text{encode}}, t_{\text{mine}}, t_{\text{rules}}$), and bounded ablation sweep harness.
- **Manuscript Layer (`paper/`)**:
  - IEEE TKDE / ACM TKDD camera-ready two-column LaTeX manuscript (`paper.tex`).
  - HTML preview builder (`build_preview.py`) and academic publication figures (`paper/figures/`).

## Feature Inventory
| # | Feature / Research Gap | Description | Milestone | Source |
|---|------------------------|-------------|-----------|--------|
| 1 | Accurate Scikit-learn Scope Framing | Issue #32712 closed as "not planned" due to supervised learning scope; external library positioning | M3 | R1, Gap 1 |
| 2 | Comprehensive Ecosystem Comparison Matrix | 8-dimension table comparing SKARM, mlxtend, scikit-mine, SPMF, and modern baselines (FIN, PrePost+, LCMFreq) | M3 | R1, Gap 2 |
| 3 | Three-Tier Evaluated Determinism | Formalize and verify result determinism ($\Delta\mathrm{supp}=0$), ordering determinism, and execution determinism | M3, M1 | R1, Gap 3 |
| 4 | Target Personas & Workflow Solutions | Practical workflow problems for Data Scientists, Data Engineers, and Model Auditors | M3 | R1, Gap 4 |
| 5 | Reframed Research Contribution | Architectural narrative centered on unified execution contracts, safe degradation, and auditable telemetry | M3 | R1, Gap 5 |
| 6 | Formal Mathematical Problem Spec | Formal definitions of $D, I, T_i \subseteq I$, support, confidence, sample weights | M1, M3 | R2, Gap 6 |
| 7 | Rigorous Data Semantics | Specifications and validation for duplicate transactions, repeated items, missing values, non-binary arrays | M1, M3 | R2, Gap 7 |
| 8 | Valid Rule Partition Constraints | Non-empty disjoint antecedent/consequent partitions $X \Rightarrow Y$ derived from $F_\sigma$ | M1, M3 | R2, Gap 8 |
| 9 | Tri-State Execution Model | Formal states: EXACT, DEGRADED, FAILED with `status_`, `is_complete_`, `termination_reason_` | M1, M3 | R2, Gap 9 |
| 10 | Candidate Count Semantics | Distinguish exact unvisited candidate counts (Apriori level $k$) from strictly unknown counts (prefix trees, future levels) | M1, M3 | R2, Gap 10 |
| 11 | Estimator API & Object Conventions | Scikit-learn parameter immutability, `clone()`, `get_params`, `set_params`, fitted attributes (`itemsets_`, `rules_`) | M1 | R3, Gap 11 |
| 12 | Pipeline Interoperability Contracts | Final estimator (`fit`/`predict`) vs intermediate transformer (`fit`/`transform`) with `get_feature_names_out()` | M1 | R3, Gap 12 |
| 13 | Unsupervised Model Selection Metrics | Scoring functions (`mean_confidence`, `mean_lift`, `total_support`) and `.score()` method for `GridSearchCV` | M1 | R3, Gap 13 |
| 14 | Algorithm Operation Counters | Detailed telemetry counters per algorithm (joins/scans in Apriori, nodes/trees in FP-Growth, intersections in Eclat) | M1 | R3, Gap 14 |
| 15 | Multi-Dimensional Resource Budgets | Enforce candidate budget ($B_{\text{cand}}$), memory budget ($B_{\text{mem}}$), and runtime budget ($B_{\text{time}}$) | M1 | R3, Gap 15 |
| 16 | Safety Governor Harmonization | Harmonize degradation thresholds and candidate generation checks across Apriori, FP-Growth, and Eclat | M1 | R3, Gap 16 |
| 17 | Symmetric Telemetry Collection | Uniform recording of wall time, CPU time, memory RSS delta, and execution metadata | M1, M2 | R4, Gap 17 |
| 18 | Explicit Research Questions (RQ1-RQ5) | Empirical evaluation structured directly around RQ1 through RQ5 | M2, M3 | R4, Gap 18 |
| 19 | 3-Stage Timing Protocol | Separate timing measurements into encoding time ($T_{\text{enc}}$), mining time ($T_{\text{mine}}$), and rule derivation ($T_{\text{rules}}$) | M2, M3 | R4, Gap 19 |
| 20 | Bounded vs Exact Ablation Sweeps | Automated evaluation of candidate caps vs itemset/rule recall, precision, and speedup trade-off curves | M2, M3 | R4, Gap 20 |
| 21 | Bug Inventory Remediation | Fix 10 cataloged bugs (B1: Zhang's metric denominator; B2: NaN tokens; B3: non-binary arrays; B4: DataFrame columns; B5-B10) | M1 | R4, Gap 21 |
| 22 | Camera-Ready LaTeX Compilation | Zero compilation errors in `paper.tex` under `miktex-pdftex.exe`, fix math unescaped `\infty` and regex text corruption | M3 | R5, Gap 22 |
| 23 | Page 1 Layout & Snippet Fix | Remove ad-hoc `\vspace*{-1.8cm}` hack, eliminate column-breaking orphan 2-line snippet in Section 1.1 | M3 | R5, Gap 23 |
| 24 | Restyle Figure 3 & Label Harmonization | Restyle `fptree.png` to academic slate-blue annotation; synchronize figure numbering sequence across paper and preview | M3 | R5, Gap 24 |
| 25 | Supervisory Authorship Block | Format Ms. Dudigam Ramya (Supervisor/Guide) and student authors from KL University with code KLCAP-2026-00340 | M3 | R5, Gap 25 |
| 26 | Citation Hygiene & References | Eliminate undefined citations and verify reference metadata for 12 bibitems | M3 | R5, Gap 26 |
| 27 | Float & Table Wrapping | Fix 104 pt overfull hbox in Table 1; eliminate longtable infinite glue shrinkage in Appendix B | M3 | R5, Gap 27 |
| 28 | Comprehensive E2E Verification Suite | Multi-tier opaque-box test suite (Tiers 1-4) verifying all requirements independently | E2E Track | Architecture |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Engine & Formal Contracts | Fix bugs B1-B10; implement formal states (`status_`, `is_complete_`), candidate count semantics, `get_feature_names_out`, `.score()`, `__sklearn_tags__`, multi-dimensional budgets, operation counters, governor harmonization | none | DONE |
| M2 | Benchmark Infrastructure & Empirical Validation | Implement 3-stage timing protocol; automated bounded ablation sweeps across dense/medium/sparse datasets; pipeline & GridSearchCV benchmarks; verify rule equivalence | M1 | DONE |
| M3 | Manuscript Revision & Publication Quality | Fix LaTeX syntax/math errors, page 1 layout/snippet, restyle Fig 3, authorship block, 8-dimension ecosystem table, RQ1-RQ5 narrative, citations, table wrapping, HTML preview | M1, M2 | DONE |
| M4 | Final Milestone & Forensic Audit | Phase 1: Pass 100% of E2E verification test suite (Tiers 1-4); Phase 2: Adversarial coverage hardening (Tier 5); Forensic integrity audit (zero violations) | M1, M2, M3, E2E | DONE |
| E2E | E2E Testing Track | Independent opaque-box test infrastructure and multi-tier test cases (Tiers 1-4) derived from user requirements; publishes TEST_READY.md | none | DONE |

## Interface Contracts
### `skarm.base` / `skarm.estimator` ↔ Scikit-learn Pipeline
- Constructor: Parameters stored without mutation (`min_support`, `min_threshold`, `metric`, `algorithm`, `max_len`, `max_candidates`, `max_time_seconds`, `max_memory_mb`, `on_candidate_explosion`, `sparse_output`, `scoring`).
- Attributes post-fit: `status_` in `{"EXACT", "DEGRADED", "FAILED"}`, `is_complete_` (bool), `is_degraded_` (bool), `termination_reason_` (Optional[str]), `itemsets_` (DataFrame), `frequent_itemsets_` (alias), `rules_` (DataFrame), `metadata_` (dict), `feature_names_in_` (ndarray), `columns_` (list).
- Methods: `fit(X, y=None, sample_weight=None)`, `predict(X)`, `transform(X)` (returns 2D binary indicator of firing rules), `get_feature_names_out(input_features=None)` (returns ndarray of rule strings), `.score(X, y=None)` (returns float).
- `__sklearn_tags__()`: Declares `requires_y=False`, `X_types=['2darray', 'sparse']`.

### Core Mining Engines (`apriori`, `fpgrowth`, `eclat`) ↔ Safety Governor (`pruning`)
- Common invocation: Each miner tracks and reports algorithm-specific operation counters:
  - Apriori: `candidates_generated`, `joins_attempted`, `scans_performed`.
  - FP-Growth: `trees_constructed`, `nodes_created`, `paths_traversed`.
  - Eclat: `intersections_performed`, `tid_elements_scanned`.
- Check governor: `pruning_policy.check_budget(step_candidates, cumulative_itemsets, elapsed_time, peak_memory)`.
- If budget exceeded with `degrade`: halts search cleanly, sets `is_degraded = True`, sets `future_candidates_status = "UNKNOWN"` (or exact unvisited for Apriori level $k$), emits `DegradedModeWarning`.

## Code Layout
- `skarm/`: Production package
  - `__init__.py`: Package entry point and version.
  - `estimator.py`: `AssociationRulesEstimator`.
  - `preprocessing.py`: `TransactionEncoder`.
  - `apriori.py`: Apriori implementation.
  - `fpgrowth.py`: FP-Growth implementation.
  - `eclat.py`: Eclat implementation.
  - `rules.py`: Association rules and metric evaluations (Zhang's metric, etc.).
  - `pruning.py`: `PruningPolicy` multi-dimensional resource governor.
  - `metrics.py`: Unsupervised scoring functions for `GridSearchCV`.
  - `telemetry.py`: Telemetry and timing logger.
  - `exceptions.py`: Custom warnings and exceptions (`DegradedModeWarning`, `CandidateExplosionError`).
- `benchmarks/`: Benchmarking suite
  - `run_pc_campaign.py`: 25-dataset benchmark campaign runner.
  - `run_bounded_ablation.py`: Automated bounded vs exact trade-off sweep.
  - `pipeline_benchmark.py`: Pipeline, ColumnTransformer, and GridSearchCV integration benchmarks.
  - `cells/`: 25 per-dataset checkpoint JSON files.
- `paper/`: Manuscript & Publication
  - `paper.tex`: LaTeX source file for IEEE TKDE / ACM TKDD camera-ready submission.
  - `build_preview.py`: HTML preview generator.
  - `figures/`: Publication-grade PNG figures.
  - `paper-preview.html`: Rendered HTML manuscript preview.
- `tests/`: Test suite
  - `test_estimator.py`, `test_apriori.py`, `test_fpgrowth.py`, `test_eclat.py`, `test_rules.py`, `test_negative_cases.py`, `run_acceptance.py`.
  - `e2e/`: Independent E2E test suite (Tiers 1-4).
