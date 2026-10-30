# HANDOFF — skarm capstone paper (KLCAP-2026-00340)

**For any LLM picking this up:** read this file first. Every claim below
has an evidence pointer, a re-verify command, and a redo instruction.
Do not trust this file blindly — re-verify the items you depend on
before building on them. Do not invent numbers, do not merge trial
records from two machines, do not push a partial result as final.

**Project:** `skarm` — Scalable Association-Rule Mining Estimator with
Apriori Baseline. Repo: `ABHISHEK1139/capstone` (main branch).
Team: Gontla Venkata Anvitha, Kamma Giridhar, Abhishek Kumar,
Paladagula Venkata Rajeev. Guide: Ms. Dudigam Ramya. University: KL
University, CSE. Commits go under Abhishek Kumar only, never the
assistant's. Nothing outward-facing without the user's per-item
go-ahead (the push of this file plus the paper draft was authorized by
the user on 30 Oct 2026).

## 1. Status dashboard (as of 30 Oct 2026, ~09:05 IST)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | 25 datasets, measured inventory | DONE | `DATASET_INVENTORY.md`; all stats parsed from files |
| 2 | Bug-fix pass, 12 files | DONE, PUSHED | commits `ff03b451`, `dd22782f`, `904b19b3` |
| 3 | Tests + acceptance harness | DONE | 25/25 tests; NT-1–NT-6 genuinely asserted |
| 4 | Dataset upload to `data/` | DONE, 26 files | all 25 dataset files + `generate_synthetic.py`; verified by repo listing 30 Oct |
| 5 | `accidents.dat`, `kosarak.dat` in git | DONE | landed with the PC push (commit `73d44df`); no longer missing |
| 6 | Benchmark campaign, VM agent | RUNNING (backup only) | 18/25 datasets, 1,280 records at last check; KNOWN correctness gap (only 12 pairs compared); do NOT merge its records into the paper |
| 7 | Benchmark campaign, PC (Windows, 12 workers) | DONE 23/25, PUSHED | commit `73d44df`: `benchmarks/campaign_results.json` (1,515 records) + `benchmarks/cells/` (23 files); missing: bms_pos, t10i4d100k |
| 8 | Threshold ladder (25 datasets) | DONE, PUSHED | `benchmarks/thresholds.json`, commit `6a76b6f` |
| 9 | Parallel PC script + sequential script fix | DONE, PUSHED | `benchmarks/run_pc_campaign.py`, commit `6a76b6f` |
| 10 | Paper, results-filled (23/25 datasets) | DONE | `paper/paper.tex` (all P1–P9 filled from cell files); `paper/paper-draft.tex` kept as the pre-results draft |
| 11 | Paper preview (readable HTML) | DONE | `paper/paper-preview.html` (regenerate via `paper/build_preview.py ../files/paper.tex`); embeds 3 measured figures |
| 12 | Gap closure: `check_estimator` | OPEN | must run; publish expected failures with reasons |
| 13 | Gap closure: pruning ablation (bounded vs exact) | OPEN | truncation counts + recall vs exact required |
| 14 | Gap closure: SPMF / modern baselines | OPEN | FIN, PrePost+, LCMFreq acknowledged, not run |
| 15 | Figures from measured data | DONE for 23/25 | `paper/figures/` runtime, memory, itemsets-vs-rules charts; regenerate after the last 2 datasets |
| 16 | Paper PDF (compiled) | OPEN | no LaTeX engine on the VM; compile on PC after the last 2 datasets land |
| 17 | Goal | ACTIVE | do NOT close until submission-ready |

## 1A. Update — 30 Oct 2026, 09:05 IST (PC results verified, paper built)

- PC campaign (Windows 11, AMD64, 16 CPUs, 12 workers, Python 3.12.10,
  sklearn 1.9.1, mlxtend 0.25.0) finished 23/25 datasets and pushed
  commit `73d44df` (08:37 IST): 1,515 records = 1,490 ok, 2 timeouts,
  8 skipped, 15 errors. Correctness: 39 cells, 145 pairs, 0 mismatches.
  30-trial protocol respected for the subset datasets present.
  Missing: bms_pos, t10i4d100k (no cell files; JSON says
  `incomplete: true`).
- SCHEMA WARNING (read before reusing the merged JSON): the merged
  `benchmarks/campaign_results.json` records contain ONLY seconds,
  throughput, status, error, n_transactions. Peak memory, itemset
  counts, rule counts, and the degraded flag live ONLY in the
  per-dataset checkpoint files `benchmarks/cells/*.json` (fields
  `peak_mem_mb`, `n_itemsets`, `n_rules`, `is_degraded`,
  `peak_candidates`, `encode_times`, `complete`). The paper's tables
  and figures were built from the cell files, not the merged JSON.
  The merged JSON is still the source for correctness pairs, skips,
  and metadata.
- Memory method correction: the campaign metadata string names a
  /proc-based sampler, but the Windows run used psutil (see
  `run_pc_campaign.py` `rss_mb()` fallback). The paper states this
  plainly; do not quote the /proc method for the PC run.
- Headline findings (all from the cell files): at each dataset's
  primary threshold (most completed skarm trials, ties to the lower
  threshold), Eclat fastest on 18 of 23, FP-Growth on 5, Apriori on
  none. Apriori timed out on accidents@0.50 and kosarak@0.005; no
  other skarm engine timed out. Degraded mode never triggered.
  mlxtend absent on 6 cells: 3 dense-cap skips (kosarak 0.01/0.005,
  kosarak_spmf 0.01) and 3 empty-input errors (docword_enron 0.5/0.7,
  pumsb_star 0.85 — skarm engines also found 0 itemsets there, so
  there is no discrepancy). `CAMPAIGN_REPORT.md` in the repo cannot
  be read through the API (pushed as a symlink); re-push it as a
  regular file. The JSON + cell files are the source of truth.
- `paper/figures/_test_scatter.png` is a valid PNG uploaded 30 Oct
  as a route test for binary pushes (base64 content round-trips
  correctly through push_files). Delete it when the real figures
  land; it is not part of the paper.
- Paper: `paper/paper.tex` fills all placeholders P1–P9 from the
  cell files (timing longtable, runtime + memory + scatter figures,
  correctness table, skips table, discussion, conclusions, machine
  spec, data/ confirmation). It states 23/25 coverage wherever a
  number depends on it and never estimates the 2 pending datasets.
  VM records are NOT merged into it (different machine, different
  ladder).
- What is really left: (1) PC re-run for bms_pos + t10i4d100k
  (`--datasets bms_pos,t10i4d100k`, resume-safe) and push; (2)
  regenerate figures + tables for 25/25; (3) compile the PDF on PC
  and check page/table overflow; (4) gap closure: check_estimator,
  pruning ablation, SPMF/modern baselines; (5) team confirmations:
  author order, 2 [VERIFY] references, scikit-mine/mlxtend re-check;
  (6) stop the VM agent once PC 25/25 is confirmed; (7) venue choice
  + submission (team decision).

## 1B. Update — 30 Oct 2026 (single LaTeX source; old files removed)

- `paper/paper.tex` (results-filled, layout-fixed, PDF compiled
  16 pp A4 on the VM) is the current paper. `paper/paper-draft.tex`
  is its regeneration source (`build_paper.py` reads it); both were
  pushed with the layout fixes (author block wrapped, architecture
  figure scaled to text width, timing table on landscape pages,
  skips-table Reason column wraps).
- REMOVED from the repo at the user's instruction:
  `paper/paper_presentation.tex` (old draft with placeholder
  authors and the invented Figure 4 formula) and
  `paper/paper_presentation.pdf` (its output). Both remain
  recoverable from git history. `paper/presentation_slides.md` was
  left in place (slides, not LaTeX code).
- `paper/paper.pdf` is NOT in the repo (too big for the MCP push
  route); the PC git-CLI push can add it.

## 2. Findings (what was wrong, missed, fixed)

### 2.1 Code bugs fixed (all pushed, all verified)
1. Eclat `max_len` off-by-one (longest itemsets silently dropped) —
   gate now `should_continue_length(len(new_prefix) + 1)` — `skarm/eclat.py`.
2. Acceptance harness printed hardcoded 100% values — now computed —
   `tests/run_acceptance.py`.
3. KPI-4 printed PASS on both branches — now follows the comparison.
4. NT-6 printed PASS without asserting — now genuinely asserted.
5. Loaders broke on accidents header line (`340183 468`), SPMF `-1`/`-2`
   tokens, BMS-WebView-1 tabs — header skipped, tokens stripped,
   delimiter sniffed — `skarm/datasets.py`.
6. Preprocessing/estimator failed on generator, DataFrame, and
   plain-string transaction inputs — all three handled.
7. CLI `--dataset` silently overridden by filename — flag now wins.
8. Rule sort unstable, contradicting the NT-5 determinism claim —
   deterministic ordering enforced — `skarm/rules.py`.
9. Benchmark KPI-1 skipped Eclat and computed recall-as-precision —
   precision AND recall for all engines.
10. Benchmark KPI-2 rubric returned 50 where its spec said "0
    otherwise" — rubric follows its spec.
11. "sklearn compliance" check was vacuous (always passed) — real check.
12. Retail Apriori `peak_ram_mb` hardcoded 0.0 — real measured peak RSS.
Verification: 25/25 tests pass; brute-force cross-checks (4 random
matrices × 2 thresholds × weighted/unweighted × 3 miners) = 0
mismatches; fixed `eclat.py` re-fetched from main, byte-identical.

### 2.2 Campaign-script bug found by smoke test (fixed, pushed)
The RSS sampler stored its stop flag in `self._stop`, shadowing an
internal `threading.Thread` method — the first trial crashed with
`TypeError: 'Event' object is not callable`. Fixed by renaming to
`self._stop_event` in both `run_pc_campaign.py` and `run_campaign.py`
(commit `6a76b6f`). This was a benchmark-script bug, not a skarm bug.

### 2.3 Windows finding (PC run)
Native Windows has no POSIX `signal.SIGALRM`, so trials on dense sets
(accidents Apriori, bms_pos Apriori) ran past the 600 s cap without
interruption. Fix applied on the PC: cross-platform async-thread
watchdog (`PyThreadState_SetAsyncExc`) that records `status: "timeout"`
and skips remaining trials of the cell, matching the Linux protocol.
Caveats: the watchdog fires when the thread next runs Python bytecode,
so a trial stuck in a long C-extension stretch can record somewhat over
600 s (acceptable — the JSON records real seconds); after an async
timeout, check the merged report for error clusters in following cells.
The watchdog-edited scripts MUST be pushed to git with the results, or
git will hold numbers from a script version it does not contain.

### 2.4 Probe findings (thresholds already proven infeasible — do not retry blindly)
- docword_nips 0.30: all four engines timed out.
- connect 0.90: Apriori timeout, FP-Growth process killed (OOM
  suspected on 7 GB RAM, NOT proven — report as "process killed", never
  "OOM proven", unless a memory-watched rerun confirms it).
- bms_pos 0.01: Apriori timeout twice, FP-Growth killed twice.
- accidents 0.50: Apriori timeout; FP-Growth OK 119.6 s; Eclat OK 278.5 s
  (8,057 itemsets, 375,774 rules); mlxtend killed.
- docword_nips 0.50: all four OK and agreeing exactly (31,582 itemsets,
  1,148,468 rules; Apriori 184.9 s, FP-Growth 73.6 s, Eclat 56.9 s,
  mlxtend 60.5 s).
- A runtime service restart (~17:26 UTC, 29 Oct 2026) killed a detached
  sweep loop once; the campaign is resume-safe from checkpoints.
- Machine limits: VM has 2 CPUs, 7 GB RAM, /tmp is a 512 MB tmpfs at
  ~99% — always set `TMPDIR` to a workspace dir and use `--no-cache-dir`
  for installs on the VM. The PC run uses 4 workers (16 GB RAM class).

### 2.5 Paper-draft problems fixed in the rewrite
Old draft (`paper/paper_presentation.tex`) had placeholder authors
(A. K. Sharma etc.), an invented Figure 4 3D surface (formula, not
measured data), padded irrelevant references, an unfair Table 4
(Retail Apriori at a different threshold), a vacuous "100% candidate
reduction" claim, only 3 datasets, a Figure 5 joining unrelated
datasets, a Figure 6 omitting Chess, dense overlapping TikZ diagrams,
and no limitations, setup, repeats, variance, or mlxtend baseline. The
new draft (`paper/paper-draft.tex`) fixes all of these; the old file is
kept in the repo for history and must NOT be used as the paper.

### 2.6 Research findings (framing — get this wrong and reviewers reject)
- scikit-learn issue #32712 ("Add Apriori", opened 14 Nov 2025) was
  CLOSED as "not_planned" on 16 Nov 2025 by maintainer @virchan on
  scope grounds, citing #2662, suggesting scikit-mine /
  scikit-learn-contrib. Linked PR #32716 also CLOSED, unmerged.
- The paper MUST frame skarm as sklearn-COMPATIBLE, never
  sklearn-endorsed or sklearn-requested. Do NOT write that sklearn
  "rejected Apriori as unimportant" — the objection was scope +
  maintenance, not merit.
- mlxtend is the honest baseline: apriori/fpgrowth/fpmax/hmine +
  full metrics (incl. conviction, Zhang), but functions-not-estimators
  and NO Eclat. Do NOT write "mlxtend is not sklearn-compatible at
  all" (it has cloneable estimators elsewhere) and do NOT quote its
  toy timing as a benchmark.
- scikit-mine exists but is dormant (no master commits after May 2023
  at last check) and lacks plain Apriori/FP-Growth/Eclat+rules; it
  mines closed/maximal/MDL patterns. Re-check at submission; do not
  call it "dead" without a fresh check.
- Secondary sources DISAGREE on dense-vs-sparse winners — the paper
  must not assert a universal winner; report matched-threshold results.
- Verified live (Python 3.12.3, scikit-learn 1.9.1):
  `from sklearn.association import Apriori` → `ModuleNotFoundError`.
  On the issue's own example (milk/bread/eggs, milk/bread,
  bread/butter) at min_sup 0.5, mlxtend gives bread 1.0, milk 0.667,
  {milk,bread} 0.667, rule {milk}→{bread} confidence 1.0 lift 1.0.
- Venues that fit: JOSS (≥6-month public-history risk for a new repo),
  SoftwareX, PeerJ Computer Science. TKDE/DMKD/KDD/ICDM are stretch.
- 12 reviewer objections with fixes are recorded in the Deep Research
  report (research notes, 29 Oct 2026) and the "Capstone Deep Research"
  page; the paper must address the fatal ones (bounded-as-exact,
  missing baselines, single-run numbers, no limitations).

### 2.7 Dataset notes (measured, not copied)
- Mushroom: stored file = 8,124 transactions (SPMF table lists 8,416)
  — report the measured figure.
- BMS-WebView-1: stored file = 59,009 transactions (commonly cited
  59,601) — report the measured figure.
- Kosarak.txt (187,140 SPMF sequences) is a DIFFERENT file from
  kosarak.dat (990,002 transactions) — never merge or conflate them.
- LFS-backed mirrors (pumsb, kosarak, T10I4D100K): the
  `raw.githubusercontent` URL returns only the LFS pointer — use the
  `media.githubusercontent` URL from the inventory.
- Licences: mirrors/UCI pages that do not state a licence are recorded
  as unknown, never guessed. Access date for all 25 datasets: 2026-10-29.
- Failed honestly, not faked: T40I10D100K, foodmart, Ta-Feng
  (Kaggle-gated only).

## 3. Protocols that must not be broken
1. **Benchmark protocol:** matched thresholds (ladder in
   `benchmarks/thresholds.json`); 30 trials for groceries, chess,
   mushroom, retail, connect, bms_pos, tic_tac_toe, docword_nips; 5
   trials otherwise; mlxtend 5 trials where a dense boolean DataFrame
   fits the 2 GB cap; 600 s per-trial cap; trial-0 timeout → remaining
   trials of that cell recorded as skipped; trial-0 correctness gate
   (itemset sets equal, supports tol 1e-9); runtime = perf_counter
   around `fit` on the pre-encoded matrix (encoding reported
   separately); memory = peak RSS sampled every 10 ms minus baseline;
   rule params confidence / min_threshold 0.25; estimator params
   max_candidates 2,000,000, sparse output, degrade on explosion.
2. **30-trial statistics are claimed ONLY for the 8-dataset subset.**
3. **Never merge trial records from two machines** into one results
   file (VM vs PC hardware differs). One campaign run = one machine,
   recorded in metadata. If the PC run finishes first, it is canonical.
4. **Paper placeholders P1–P9** (in `paper/paper-draft.tex`) are filled
   ONLY from `benchmarks/campaign_results.json` after the campaign
   prints CAMPAIGN COMPLETE. Never fill by hand, never copy from the
   old draft, never push a partial results file as final (a partial
   commit MUST say PARTIAL + missing datasets).
5. **No invented numbers anywhere** — timeouts, skips, mismatches, and
   degraded runs are reported with reasons, never hidden or smoothed.
6. **Portal contract:** objectives O1–O5; held-out partition reserved
   before tuning (AC-3); NT-1–NT-6; KPI-4 = 30 trials with p50/p95/p99
   and p95 ≤ 0.8× baseline; KPI-2 = rubric ≥80/100 frozen before O2
   with two independent raters; limitations must state excluded scope
   (no production certification, no universal generalisation, no
   unattended high-stakes operation); no sponsor is stated — do not
   invent one; do NOT repeat the portal's "Aerospace & Space" sector
   label for this software library.

## 4. Re-verify checklist (run these before building on anything)
- [ ] `data/` listing = 24 files (19 new + 5 originals); only
      accidents.dat and kosarak.dat missing. Re-fetch via the GitHub
      API contents listing and compare names.
- [ ] Fixed `eclat.py` on main is byte-identical to the fixed local
      copy (re-fetch and diff).
- [ ] Tests: run the acceptance harness → expect 25/25, NT-1–NT-6
      asserted (not hardcoded).
- [ ] Smoke test the PC script on 2 tiny datasets
      (`--datasets tic_tac_toe,balance_scale --workers 2
      --skip-mlxtend`) → both `done`, all records `ok`, correctness
      `match=True`; then delete `benchmarks/cells`,
      `campaign_results.json`, `CAMPAIGN_REPORT.md` before a full run.
- [ ] Threshold ladder: `benchmarks/thresholds.json` has 25 datasets;
      every `expected_n` matches `DATASET_INVENTORY.md`.
- [ ] Paper draft: every `\cite{}` key has a `\bibitem`, every
      `\bibitem` is cited; no placeholder authors; no invented 3D
      formula; exactly the P1–P9 placeholders remain empty.
- [ ] After campaign merge: `CAMPAIGN COMPLETE` printed; no missing
      datasets; every correctness line `match=True` or explained;
      every skip has a reason; 30-trial stats only for the subset.

## 5. Redo instructions (if a verification fails)
- **A dataset file is wrong/missing:** re-download from the inventory
  URL, verify the count by parsing, and push via git CLI (for the 2
  big files) — never via the browser UI for files over 25 MB, and
  never upload a duplicate without checking the `data/` listing first.
- **A code fix regressed:** re-apply from the fix commits listed in
  §2.1; re-run the acceptance harness; do not reintroduce hardcoded
  harness values.
- **Campaign cell failed/killed:** delete only that dataset's
  checkpoint file (`benchmarks/cells/<dataset>.json`) and re-run the
  same command — the campaign resumes from all other checkpoints.
  Lower `--workers` if the failure was memory. Do NOT loop a killed
  cell hoping it survives; record it as a skip with evidence.
- **Campaign script crash on Windows:** the SIGALRM gap is fixed by
  the async-thread watchdog (§2.3); if it still fails, run inside
  WSL2 instead and record which mechanism was actually used.
- **VM campaign stalls:** it is the backup. If it produces no output
  for over 2 hours, prefer the PC run; do not merge the two.
- **Paper placeholders filled with wrong numbers:** revert to the
  placeholder version and refill strictly from
  `campaign_results.json`; check the per-cell table against the JSON
  record by record.

## 6. Open items needing the user (do not decide these alone)
- Confirm author order on the paper before submission.
- Verify before submission: Fournier-Viger et al. 2022 LNCS page
  numbers; Kallay & Mihoc 2025 venue; scikit-mine activity;
  mlxtend version.
- Decide packing/support staff role limits and staging-first schema
  apply for the separate RK Saree project (unrelated to this paper).
- Per-item go-ahead is still required for any new outward submission.

## 7. File map
- Repo: `skarm/` (library), `benchmarks/` (scripts + thresholds +
  results after merge), `data/` (datasets), `tests/` (acceptance
  harness), `docs/D1–D7.md` (portal deliverables),
  `paper/paper_presentation.tex` (OLD draft — history only),
  `paper/paper-draft.tex` (NEW draft), `paper/paper-preview.html`
  (readable preview), `paper/build_preview.py` (preview builder),
  `DATASET_INVENTORY.md`, `HANDOFF.md` (this file).
- Goal workspace (not in git): `paper-preview.html` source build
  script copy, PC benchmark prompt, Deep Research page and report.
- The VM mirror (`capstone-mirror/`) is scratch: code files, `data_new/`
  (local dataset copies), probe logs. The repo is the source of truth.

## 8. What "done" means for this goal
The goal closes ONLY when: benchmarks merged from one machine with all
checks in §4 passing; gap closure (check_estimator, pruning ablation)
run and reported; figures rebuilt from measured data and visually
verified; paper placeholders P1–P9 filled from the JSON; PDF compiled;
and the paper is submission-ready. Until then the goal stays ACTIVE.
