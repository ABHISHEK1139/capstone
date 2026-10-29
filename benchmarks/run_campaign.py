"""Full benchmark campaign for the skarm paper.

Reads thresholds + rationale from benchmarks/thresholds.json (finalized from
the probe phase). For each dataset x threshold x engine runs N trials
(30 for the 8-dataset subset, 5 otherwise; mlxtend always 5 where feasible),
recording runtime, peak RSS (sampled every 10 ms, baseline-subtracted),
n_itemsets, n_rules, throughput, degraded flag, peak candidates.

Per-trial cap: 600 s via SIGALRM. If trial 1 of a cell times out, the
remaining trials of that cell are skipped with the reason recorded.
Correctness: on trial 1 of each cell, skarm engines must agree exactly on
itemsets (keys) and supports (tolerance 1e-9); mlxtend is compared by name
where it ran. Verdicts are recorded per cell, never smoothed over.

Writes benchmarks/campaign_results.json incrementally after each cell.
No skarm code is modified by this script. Resume-safe: completed cells in
an existing campaign_results.json are skipped.
"""
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_new")
THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "thresholds.json")
OUT = os.path.join(os.path.dirname(__file__), "campaign_results.json")

TRIAL_TIMEOUT_S = 600
MLXTEND_TRIALS = 5
MLXTEND_DENSE_BYTE_CAP = 2_000_000_000
ENGINES = ["apriori", "fpgrowth", "eclat"]
# 30-trial protocol subset (portal KPI contract).
SUBSET_30 = ["groceries", "chess", "mushroom", "retail", "connect",
             "bms_pos", "tic_tac_toe", "docword_nips"]


class RunTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise RunTimeout("exceeded 600 s trial cap")


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
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


def timed_trial(fn):
    """Run fn() with SIGALRM cap + RSS sampling. Returns record dict."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TRIAL_TIMEOUT_S)
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
        return {"status": "timeout", "seconds": round(secs, 4), "error": str(e)}
    except Exception as e:  # noqa: BLE001 - campaign must record, not crash
        secs = time.perf_counter() - t0
        sampler.stop()
        return {"status": "error", "seconds": round(secs, 4),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


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


def main():
    with open(THRESHOLDS_PATH) as f:
        cfg = json.load(f)
    datasets_cfg = cfg["datasets"]

    import importlib.metadata
    def ver(pkg):
        try:
            return importlib.metadata.version(pkg)
        except Exception:
            return None

    results = {
        "metadata": {
            "platform": f"{platform.machine()}, {platform.system()} {platform.release()}",
            "cpu_count": os.cpu_count(),
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
                             "minus pre-trial baseline RSS",
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
    }
    if os.path.exists(OUT):
        with open(OUT) as f:
            results = json.load(f)
    done_cells = set()
    for r in results["records"]:
        done_cells.add((r["dataset"], str(r["threshold"]), r["engine"], r["trial"]))
    # A cell is complete when all its planned trials are recorded ok/timeout/error.
    def cell_done(name, thr, eng, n_trials):
        return all((name, str(thr), eng, t) in done_cells for t in range(n_trials))

    for name, dcfg in datasets_cfg.items():
        fname = dcfg["file"]
        thresholds = dcfg["thresholds"]
        n_trials = 30 if name in SUBSET_30 else 5
        print(f"=== {name} ({fname}) trials={n_trials} ===", flush=True)
        path = os.path.join(DATA_DIR, fname)
        try:
            tx = load_transactions(path)
        except Exception as e:  # noqa: BLE001
            results["skips"].append({"dataset": name, "reason": f"load failed: {e}"})
            print(f"  LOAD FAILED: {e}", flush=True)
            continue
        if len(tx) != dcfg["expected_n"]:
            results["skips"].append(
                {"dataset": name,
                 "reason": f"parsed count {len(tx)} != inventory {dcfg['expected_n']}"})
            print("  COUNT MISMATCH — skipped", flush=True)
            continue
        t0 = time.perf_counter()
        encoder = TransactionEncoder(sparse_output=True)
        X = encoder.fit_transform(tx)
        encode_sec = round(time.perf_counter() - t0, 3)
        columns = [str(c) for c in encoder.columns_]
        del tx
        print(f"  encoded {X.shape} in {encode_sec}s", flush=True)
        dense_bytes = int(X.shape[0]) * int(X.shape[1])
        mlxtend_feasible = dense_bytes <= MLXTEND_DENSE_BYTE_CAP

        for thr in thresholds:
            # --- trial 1 for all engines first (correctness gate) ---
            trial1_itemsets = {}
            for eng in ENGINES:
                if cell_done(name, thr, eng, n_trials):
                    print(f"  thr={thr} {eng}: already complete, skipping", flush=True)
                    continue

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
                results["records"].append(rec)
                print(f"  thr={thr} {eng} trial0: {rec['status']} {rec['seconds']}s "
                      f"itemsets={rec.get('n_itemsets')} rules={rec.get('n_rules')} "
                      f"degraded={rec.get('is_degraded')}", flush=True)
                with open(OUT, "w") as f:
                    json.dump(results, f, indent=2)
                if rec["status"] == "timeout":
                    for t in range(1, n_trials):
                        results["records"].append(
                            {"dataset": name, "threshold": thr, "engine": eng, "trial": t,
                             "status": "skipped", "seconds": 0,
                             "error": "skipped after trial-0 timeout (600 s cap)"})
                    results["skips"].append(
                        {"dataset": name, "threshold": thr, "engine": eng,
                         "reason": "trial 0 exceeded 600 s; remaining trials skipped"})
                    with open(OUT, "w") as f:
                        json.dump(results, f, indent=2)

            # --- correctness verdict for this cell ---
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
            results["correctness"][cell_key] = verdict

            # --- remaining trials for engines that did not time out ---
            for eng in ENGINES:
                if cell_done(name, thr, eng, n_trials):
                    continue
                if any(r["dataset"] == name and str(r["threshold"]) == str(thr)
                       and r["engine"] == eng and r["trial"] == 0
                       and r["status"] == "timeout" for r in results["records"]):
                    continue
                for t in range(1, n_trials):
                    if (name, str(thr), eng, t) in done_cells:
                        continue

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
                    results["records"].append(rec)
                    with open(OUT, "w") as f:
                        json.dump(results, f, indent=2)
                print(f"  thr={thr} {eng}: trials 1..{n_trials - 1} done", flush=True)

            # --- mlxtend baseline ---
            if cell_done(name, thr, "mlxtend", MLXTEND_TRIALS):
                print(f"  thr={thr} mlxtend: already complete, skipping", flush=True)
            elif not mlxtend_feasible:
                results["skips"].append(
                    {"dataset": name, "threshold": thr, "engine": "mlxtend",
                     "reason": f"dense boolean DataFrame would need ~{dense_bytes / 1e9:.1f} GB"})
                print(f"  thr={thr} mlxtend: skipped (dense DataFrame ~{dense_bytes / 1e9:.1f} GB)",
                      flush=True)
            else:
                ml_itemsets_named = None
                for t in range(MLXTEND_TRIALS):
                    if (name, str(thr), "mlxtend", t) in done_cells:
                        continue

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
                    results["records"].append(rec)
                    with open(OUT, "w") as f:
                        json.dump(results, f, indent=2)
                    if rec["status"] == "timeout":
                        for tt in range(t + 1, MLXTEND_TRIALS):
                            results["records"].append(
                                {"dataset": name, "threshold": thr, "engine": "mlxtend",
                                 "trial": tt, "status": "skipped", "seconds": 0,
                                 "error": "skipped after timeout (600 s cap)"})
                        results["skips"].append(
                            {"dataset": name, "threshold": thr, "engine": "mlxtend",
                             "reason": f"trial {t} exceeded 600 s; remaining trials skipped"})
                        with open(OUT, "w") as f:
                            json.dump(results, f, indent=2)
                        break
                if ml_itemsets_named is not None and "apriori" in trial1_itemsets:
                    named_skarm = {frozenset(columns[i] for i in k): v
                                   for k, v in trial1_itemsets["apriori"].items()}
                    results["correctness"][cell_key]["skarm_apriori_vs_mlxtend"] = \
                        compare_itemsets(named_skarm, ml_itemsets_named)
                print(f"  thr={thr} mlxtend: done", flush=True)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)

        # per-dataset encode time record
        results.setdefault("encode_times", {})[name] = encode_sec
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
