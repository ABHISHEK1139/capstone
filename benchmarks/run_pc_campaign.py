"""Parallel benchmark campaign for the skarm paper — PC version.

Same protocol as the sequential campaign (benchmarks/run_campaign.py):
matched thresholds from benchmarks/thresholds.json, 30 trials for the
8-dataset portal subset, 5 trials otherwise, mlxtend apriori baseline
(5 trials where a dense boolean DataFrame fits the 2 GB cap), trial-0
correctness gate (all skarm engines must agree exactly on itemsets and
supports, tol 1e-9), 600 s per-trial cap via SIGALRM, trial-0 timeout
skips the remaining trials of that cell. Nothing is faked: timeouts,
errors and skips are recorded with reasons.

Difference from the sequential version: datasets run in parallel in a
process pool (one worker per dataset task, encoded matrix built once per
dataset inside the worker). Each dataset writes its own checkpoint file
benchmarks/cells/<dataset>.json (atomic rename); a dataset with a
complete checkpoint is skipped on re-runs, so the campaign is
resume-safe. Final assembly merges all checkpoints into
benchmarks/campaign_results.json in the SAME format as the sequential
campaign, and --summarize builds benchmarks/CAMPAIGN_REPORT.md.

Requirements: Linux, macOS (RSS reads 0.0 without /proc), or WSL2 on
Windows. Native Windows is NOT supported (SIGALRM does not exist there).
Python 3.10+ with scikit-learn, mlxtend, numpy, pandas, scipy installed.

Usage:
  python benchmarks/run_pc_campaign.py --workers 8
  python benchmarks/run_pc_campaign.py --datasets tic_tac_toe,balance_scale --workers 2 --skip-mlxtend
  python benchmarks/run_pc_campaign.py --assemble-only --summarize
  python benchmarks/run_pc_campaign.py --summarize   # report from existing campaign_results.json

To retry one dataset after fixing its cause (e.g. a missing file):
  rm benchmarks/cells/<dataset>.json
  python benchmarks/run_pc_campaign.py --datasets <dataset>

Do NOT merge records from two machines into one results file: runtimes
from different hardware are not comparable. One campaign run = one
machine, recorded in metadata.
"""
import argparse
import json
import os
import platform
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skarm.datasets import load_transactions
from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator

BASE = os.path.dirname(__file__)
DEFAULT_DATA_DIR = os.path.join(BASE, "..", "data")
THRESHOLDS_PATH = os.path.join(BASE, "thresholds.json")
CELLS_DIR = os.path.join(BASE, "cells")
OUT = os.path.join(BASE, "campaign_results.json")
REPORT = os.path.join(BASE, "CAMPAIGN_REPORT.md")

TRIAL_TIMEOUT_S = 600
MLXTEND_TRIALS = 5
MLXTEND_DENSE_BYTE_CAP = 2_000_000_000
ENGINES = ["apriori", "fpgrowth", "eclat"]
# 30-trial protocol subset (portal KPI contract). Paper claims 30-trial
# statistics ONLY for these datasets.
SUBSET_30 = ["groceries", "chess", "mushroom", "retail", "connect",
             "bms_pos", "tic_tac_toe", "docword_nips"]

DATA_DIR = DEFAULT_DATA_DIR
SKIP_MLXTEND = False


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout("exceeded 600 s trial cap")


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


class RssSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.max_rss = 0.0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                r = rss_mb()
                if r > self.max_rss:
                    self.max_rss = r
            except Exception:
                pass
            time.sleep(0.01)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)


HAS_SIGALRM = hasattr(signal, "SIGALRM")

def timed_trial(fn):
    """Run fn() with 600 s cap (SIGALRM on POSIX, thread watchdog via PyThreadState_SetAsyncExc on Windows) + RSS sampling."""
    old = None
    cancel_event = None
    if HAS_SIGALRM:
        old = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(TRIAL_TIMEOUT_S)
    else:
        import ctypes
        cancel_event = threading.Event()
        target_tid = threading.get_native_id()
        def _watchdog_fn():
            if not cancel_event.wait(TRIAL_TIMEOUT_S):
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(target_tid),
                    ctypes.py_object(RunTimeout)
                )
        watchdog = threading.Thread(target=_watchdog_fn, daemon=True)
        watchdog.start()

    baseline = rss_mb()
    sampler = RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        res = fn()
        secs = time.perf_counter() - t0
        sampler.stop()
        peak = max(sampler.max_rss - baseline, 0.0)
        rec = {"status": "ok", "seconds": round(secs, 4),
               "peak_mem_mb": round(peak, 2)}
        rec.update(res)
        return rec
    except RunTimeout as e:
        secs = time.perf_counter() - t0
        sampler.stop()
        return {"status": "timeout", "seconds": round(secs, 4), "error": str(e) or "exceeded 600 s trial cap"}
    except Exception as e:  # noqa: BLE001 - campaign must record, not crash
        secs = time.perf_counter() - t0
        sampler.stop()
        return {"status": "error", "seconds": round(secs, 4),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        if HAS_SIGALRM:
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)
        elif cancel_event is not None:
            cancel_event.set()


def compare_itemsets(a, b, tol=1e-9):
    """Compare two {frozenset: support} dicts. Returns verdict dict."""
    ka, kb = set(a.keys()), set(b.keys())
    if ka != kb:
        return {"match": False, "reason": "key sets differ",
                "only_in_a": len(ka - kb), "only_in_b": len(kb - ka),
                "n_a": len(ka), "n_b": len(kb), "max_support_diff": None}
    max_diff = 0.0
    for k in ka:
        d = abs(a[k] - b[k])
        if d > max_diff:
            max_diff = d
    return {"match": max_diff <= tol, "reason": None if max_diff <= tol else "support differs",
            "only_in_a": 0, "only_in_b": 0, "n_a": len(ka), "n_b": len(kb),
            "max_support_diff": max_diff}


def run_dataset(task):
    """Worker: run one whole dataset (all thresholds, all engines).

    Returns a short status string for the parent's progress log. The full
    per-dataset bundle is written to cells/<dataset>.json.
    """
    global DATA_DIR, SKIP_MLXTEND
    name, dcfg, DATA_DIR, SKIP_MLXTEND = task
    fname = dcfg["file"]
    thresholds = dcfg["thresholds"]
    n_trials = 30 if name in SUBSET_30 else 5
    bundle = {"dataset": name, "records": [], "correctness": {}, "skips": [],
              "encode_times": {}, "complete": False}
    path = os.path.join(DATA_DIR, fname)
    try:
        tx = load_transactions(path)
    except Exception as e:  # noqa: BLE001
        bundle["skips"].append({"dataset": name, "reason": f"load failed: {e}"})
        bundle["complete"] = True
        _write_cell(name, bundle)
        return f"{name}: LOAD FAILED: {e}"
    if len(tx) != dcfg["expected_n"]:
        bundle["skips"].append(
            {"dataset": name,
             "reason": f"parsed count {len(tx)} != inventory {dcfg['expected_n']}"})
        bundle["complete"] = True
        _write_cell(name, bundle)
        return f"{name}: COUNT MISMATCH ({len(tx)} != {dcfg['expected_n']}), skipped"
    t0 = time.perf_counter()
    encoder = TransactionEncoder(sparse_output=True)
    X = encoder.fit_transform(tx)
    encode_sec = round(time.perf_counter() - t0, 3)
    columns = [str(c) for c in encoder.columns_]
    del tx
    dense_bytes = int(X.shape[0]) * int(X.shape[1])
    mlxtend_feasible = dense_bytes <= MLXTEND_DENSE_BYTE_CAP

    for thr in thresholds:
        trial1_itemsets = {}
        for eng in ENGINES:
            def run_eng(_eng=eng):
                est = AssociationRulesEstimator(
                    min_support=thr, metric="confidence", min_threshold=0.25,
                    algorithm=_eng, max_candidates=2_000_000, sparse_output=True,
                )
                est.fit(X)
                return {
                    "n_itemsets": len(est.frequent_itemsets_),
                    "n_rules": len(est.rules_),
                    "is_degraded": bool(est.is_degraded_),
                    "peak_candidates": int(est.pruning_policy_.peak_candidate_count),
                    "_itemsets": est.frequent_itemsets_,
                }
            rec = timed_trial(run_eng)
            itemsets = rec.pop("_itemsets", None)
            if itemsets is not None:
                trial1_itemsets[eng] = itemsets
            rec.update({"dataset": name, "threshold": thr, "engine": eng, "trial": 0,
                        "n_transactions": int(X.shape[0]),
                        "throughput": round(X.shape[0] / rec["seconds"], 1)
                        if rec["status"] == "ok" and rec["seconds"] > 0 else 0})
            bundle["records"].append(rec)
            if rec["status"] == "timeout":
                for t in range(1, n_trials):
                    bundle["records"].append(
                        {"dataset": name, "threshold": thr, "engine": eng, "trial": t,
                         "status": "skipped", "seconds": 0,
                         "error": "skipped after trial-0 timeout (600 s cap)"})
                bundle["skips"].append(
                    {"dataset": name, "threshold": thr, "engine": eng,
                     "reason": "trial 0 exceeded 600 s; remaining trials skipped"})

        cell_key = f"{name}@{thr}"
        verdict = {}
        engs = [e for e in ENGINES if e in trial1_itemsets]
        if len(engs) >= 2:
            for i in range(len(engs)):
                for j in range(i + 1, len(engs)):
                    verdict[f"{engs[i]}_vs_{engs[j]}"] = compare_itemsets(
                        trial1_itemsets[engs[i]], trial1_itemsets[engs[j]])
        else:
            verdict["note"] = f"only {len(engs)} engine(s) completed trial 0; no comparison"
        bundle["correctness"][cell_key] = verdict

        for eng in ENGINES:
            if any(r["dataset"] == name and str(r["threshold"]) == str(thr)
                   and r["engine"] == eng and r["trial"] == 0
                   and r["status"] == "timeout" for r in bundle["records"]):
                continue
            for t in range(1, n_trials):
                def run_eng2(_eng=eng):
                    est = AssociationRulesEstimator(
                        min_support=thr, metric="confidence", min_threshold=0.25,
                        algorithm=_eng, max_candidates=2_000_000, sparse_output=True,
                    )
                    est.fit(X)
                    return {
                        "n_itemsets": len(est.frequent_itemsets_),
                        "n_rules": len(est.rules_),
                        "is_degraded": bool(est.is_degraded_),
                        "peak_candidates": int(est.pruning_policy_.peak_candidate_count),
                    }
                rec = timed_trial(run_eng2)
                rec.update({"dataset": name, "threshold": thr, "engine": eng, "trial": t,
                            "n_transactions": int(X.shape[0]),
                            "throughput": round(X.shape[0] / rec["seconds"], 1)
                            if rec["status"] == "ok" and rec["seconds"] > 0 else 0})
                bundle["records"].append(rec)

        if SKIP_MLXTEND:
            bundle["skips"].append(
                {"dataset": name, "threshold": thr, "engine": "mlxtend",
                 "reason": "skipped by --skip-mlxtend flag (smoke test / no mlxtend installed)"})
        elif not mlxtend_feasible:
            bundle["skips"].append(
                {"dataset": name, "threshold": thr, "engine": "mlxtend",
                 "reason": f"dense boolean DataFrame would need ~{dense_bytes / 1e9:.1f} GB"})
        else:
            ml_itemsets_named = None
            for t in range(MLXTEND_TRIALS):
                def run_ml():
                    import pandas as pd
                    from mlxtend.frequent_patterns import apriori as ml_apriori
                    from mlxtend.frequent_patterns import association_rules as ml_rules
                    df = pd.DataFrame(X.toarray(), columns=columns, dtype=bool)
                    freq = ml_apriori(df, min_support=thr, use_colnames=True)
                    rules = ml_rules(freq, metric="confidence", min_threshold=0.25)
                    named = {frozenset(r): float(s) for r, s in
                             zip(freq["itemsets"], freq["support"])}
                    return {"n_itemsets": len(freq), "n_rules": len(rules),
                            "_named": named}
                rec = timed_trial(run_ml)
                named = rec.pop("_named", None)
                if named is not None and ml_itemsets_named is None:
                    ml_itemsets_named = named
                rec.update({"dataset": name, "threshold": thr, "engine": "mlxtend",
                            "trial": t, "n_transactions": int(X.shape[0]),
                            "throughput": round(X.shape[0] / rec["seconds"], 1)
                            if rec["status"] == "ok" and rec["seconds"] > 0 else 0})
                bundle["records"].append(rec)
                if rec["status"] == "timeout":
                    for tt in range(t + 1, MLXTEND_TRIALS):
                        bundle["records"].append(
                            {"dataset": name, "threshold": thr, "engine": "mlxtend",
                             "trial": tt, "status": "skipped", "seconds": 0,
                             "error": "skipped after timeout (600 s cap)"})
                    bundle["skips"].append(
                        {"dataset": name, "threshold": thr, "engine": "mlxtend",
                         "reason": f"trial {t} exceeded 600 s; remaining trials skipped"})
                    break
            if ml_itemsets_named is not None and "apriori" in trial1_itemsets:
                named_skarm = {frozenset(columns[i] for i in k): v
                               for k, v in trial1_itemsets["apriori"].items()}
                bundle["correctness"][cell_key]["skarm_apriori_vs_mlxtend"] = \
                    compare_itemsets(named_skarm, ml_itemsets_named)

    bundle["encode_times"][name] = encode_sec
    bundle["complete"] = True
    _write_cell(name, bundle)
    n_ok = sum(1 for r in bundle["records"] if r["status"] == "ok")
    return f"{name}: done ({len(bundle['records'])} records, {n_ok} ok, " \
           f"{len(bundle['skips'])} skips, encode {encode_sec}s)"


def _write_cell(name, bundle):
    os.makedirs(CELLS_DIR, exist_ok=True)
    tmp = os.path.join(CELLS_DIR, f".{name}.json.tmp")
    final = os.path.join(CELLS_DIR, f"{name}.json")
    with open(tmp, "w") as f:
        json.dump(bundle, f, indent=2)
    os.replace(tmp, final)


def assemble(workers):
    """Merge cells/*.json into campaign_results.json (original format)."""
    import importlib.metadata

    def ver(pkg):
        try:
            return importlib.metadata.version(pkg)
        except Exception:
            return None

    with open(THRESHOLDS_PATH) as f:
        cfg = json.load(f)
    datasets_cfg = cfg["datasets"]
    results = {
        "metadata": {
            "platform": f"{platform.machine()}, {platform.system()} {platform.release()}",
            "cpu_count": os.cpu_count(),
            "workers": workers,
            "run_mode": "pc-parallel (one worker per dataset)",
            "python_version": sys.version.split()[0],
            "scikit_learn_version": ver("scikit-learn"),
            "mlxtend_version": ver("mlxtend"),
            "numpy_version": ver("numpy"),
            "pandas_version": ver("pandas"),
            "scipy_version": ver("scipy"),
            "rule_params": {"metric": "confidence", "min_threshold": 0.25},
            "estimator_params": {"max_candidates": 2_000_000, "sparse_output": True,
                                 "on_candidate_explosion": "degrade"},
            "trial_timeout_s": TRIAL_TIMEOUT_S,
            "memory_method": "peak RSS sampled every 10 ms from /proc/self/status VmRSS, "
                             "minus pre-trial baseline RSS (0.0 where /proc is unavailable)",
            "runtime_method": "time.perf_counter around AssociationRulesEstimator.fit on a "
                             "pre-encoded matrix (encoding time reported separately per dataset); "
                             "mlxtend runtime covers TransactionEncoder-equivalent dense DataFrame "
                             "build + apriori + association_rules",
            "trial_counts": {"subset_30": SUBSET_30, "other": 5, "mlxtend": MLXTEND_TRIALS},
            "threshold_rationale": {k: v["rationale"] for k, v in datasets_cfg.items()},
        },
        "records": [],
        "correctness": {},
        "skips": [],
        "encode_times": {},
    }
    missing = []
    for name in datasets_cfg:
        cell_path = os.path.join(CELLS_DIR, f"{name}.json")
        if not os.path.exists(cell_path):
            missing.append(name)
            continue
        with open(cell_path) as f:
            bundle = json.load(f)
        if not bundle.get("complete"):
            missing.append(name)
            continue
        results["records"].extend(bundle["records"])
        results["correctness"].update(bundle["correctness"])
        results["skips"].extend(bundle["skips"])
        results["encode_times"].update(bundle["encode_times"])
    results["incomplete"] = bool(missing)
    results["missing_datasets"] = missing
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    return results, missing


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    import math
    k = math.ceil(pct / 100.0 * len(sorted_vals)) - 1
    return sorted_vals[max(0, min(k, len(sorted_vals) - 1))]


def summarize():
    """Build CAMPAIGN_REPORT.md from campaign_results.json."""
    with open(OUT) as f:
        results = json.load(f)
    meta = results["metadata"]
    cells = {}
    for r in results["records"]:
        cells.setdefault((r["dataset"], r["threshold"], r["engine"]), []).append(r)
    lines = []
    lines.append("# skarm benchmark campaign report (PC run)")
    lines.append("")
    if results.get("incomplete"):
        lines.append(f"> PARTIAL — missing datasets: {', '.join(results['missing_datasets'])}. "
                     "Do not use this report for paper claims until the run completes.")
    else:
        lines.append("> COMPLETE — all datasets in thresholds.json have a checkpoint file.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for k in ("platform", "cpu_count", "workers", "run_mode", "python_version",
              "scikit_learn_version", "mlxtend_version", "numpy_version",
              "pandas_version", "scipy_version", "trial_timeout_s"):
        lines.append(f"- {k}: {meta.get(k)}")
    lines.append(f"- rule params: {meta['rule_params']}")
    lines.append(f"- estimator params: {meta['estimator_params']}")
    lines.append(f"- memory method: {meta['memory_method']}")
    lines.append(f"- runtime method: {meta['runtime_method']}")
    lines.append("")
    lines.append("## Per-cell results (seconds; mean / p50 / p95 over OK trials only)")
    lines.append("")
    lines.append("| dataset | threshold | engine | ok trials | mean | p50 | p95 | mean peak MB | itemsets | rules | degraded |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for (ds, thr, eng) in sorted(cells, key=lambda x: (x[0], float(x[1]), x[2])):
        recs = cells[(ds, thr, eng)]
        ok = sorted(r["seconds"] for r in recs if r["status"] == "ok")
        mem = [r["peak_mem_mb"] for r in recs if r["status"] == "ok"]
        t0 = next((r for r in recs if r["trial"] == 0), None)
        mean = round(sum(ok) / len(ok), 3) if ok else 0
        lines.append(f"| {ds} | {thr} | {eng} | {len(ok)}/{len(recs)} | {mean} | "
                     f"{round(_percentile(ok, 50), 3)} | {round(_percentile(ok, 95), 3)} | "
                     f"{round(sum(mem) / len(mem), 1) if mem else 0} | "
                     f"{t0.get('n_itemsets', '-') if t0 else '-'} | "
                     f"{t0.get('n_rules', '-') if t0 else '-'} | "
                     f"{t0.get('is_degraded', '-') if t0 else '-'} |")
    lines.append("")
    lines.append("## Correctness (trial-0 itemset agreement, tol 1e-9)")
    lines.append("")
    for cell_key in sorted(results["correctness"]):
        verdict = results["correctness"][cell_key]
        for pair, v in verdict.items():
            if pair == "note":
                lines.append(f"- {cell_key}: {v}")
            else:
                lines.append(f"- {cell_key} {pair}: match={v['match']} "
                             f"(n_a={v['n_a']}, n_b={v['n_b']}, "
                             f"max_support_diff={v['max_support_diff']}, reason={v['reason']})")
    lines.append("")
    lines.append("## Skips, timeouts and errors (recorded, never hidden)")
    lines.append("")
    if not results["skips"]:
        lines.append("- none")
    for s in results["skips"]:
        lines.append(f"- {s.get('dataset')} {s.get('threshold', '')} "
                     f"{s.get('engine', '')}: {s['reason']}")
    lines.append("")
    lines.append("## Threshold rationale")
    lines.append("")
    for name, rationale in meta["threshold_rationale"].items():
        lines.append(f"- {name}: {rationale}")
    lines.append("")
    lines.append("## Trial-count rule (read before citing)")
    lines.append("")
    lines.append("30-trial statistics (p50/p95/mean) are claimed ONLY for the subset: "
                 + ", ".join(SUBSET_30) + ". All other datasets report 5 trials; "
                 "mlxtend always reports 5 trials where feasible.")
    lines.append("")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    return REPORT


def main():
    global DATA_DIR, SKIP_MLXTEND
    ap = argparse.ArgumentParser(description="Parallel skarm benchmark campaign (PC run)")
    ap.add_argument("--workers", type=int, default=max(2, min(12, (os.cpu_count() or 4) - 2)),
                    help="process-pool workers (default: cpu_count-2, clamped 2..12)")
    ap.add_argument("--datasets", default="",
                    help="comma-separated dataset keys to run (default: all in thresholds.json)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="folder holding the dataset files (default: repo data/)")
    ap.add_argument("--skip-mlxtend", action="store_true",
                    help="skip the mlxtend baseline (smoke tests / mlxtend not installed)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="only merge existing cells/*.json into campaign_results.json")
    ap.add_argument("--summarize", action="store_true",
                    help="build CAMPAIGN_REPORT.md from campaign_results.json")
    args = ap.parse_args()
    DATA_DIR = args.data_dir
    SKIP_MLXTEND = args.skip_mlxtend

    with open(THRESHOLDS_PATH) as f:
        cfg = json.load(f)
    datasets_cfg = cfg["datasets"]
    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()] or list(datasets_cfg)
    unknown = [d for d in wanted if d not in datasets_cfg]
    if unknown:
        print(f"Unknown datasets (check thresholds.json keys): {unknown}", flush=True)
        return

    if not args.assemble_only:
        tasks = []
        for name in wanted:
            cell_path = os.path.join(CELLS_DIR, f"{name}.json")
            if os.path.exists(cell_path):
                with open(cell_path) as f:
                    if json.load(f).get("complete"):
                        print(f"{name}: checkpoint complete, skipping", flush=True)
                        continue
            tasks.append((name, datasets_cfg[name], DATA_DIR, SKIP_MLXTEND))
        if tasks:
            from multiprocessing import Pool
            print(f"Running {len(tasks)} dataset(s) on {args.workers} worker(s) "
                  f"(data: {DATA_DIR})", flush=True)
            with Pool(processes=args.workers) as pool:
                for status in pool.imap_unordered(run_dataset, tasks):
                    print(f"  {status}", flush=True)
        else:
            print("Nothing to run (all requested datasets have complete checkpoints).", flush=True)

    results, missing = assemble(args.workers)
    print(f"Merged {len(results['records'])} records into campaign_results.json", flush=True)
    if missing:
        print(f"INCOMPLETE — still missing: {', '.join(missing)}", flush=True)
    else:
        print("CAMPAIGN COMPLETE", flush=True)
    if args.summarize:
        print(f"Report written: {summarize()}", flush=True)


if __name__ == "__main__":
    main()
