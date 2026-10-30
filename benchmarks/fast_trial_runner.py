"""Trial-parallel benchmark runner for remaining datasets (bms_pos, t10i4d100k).
Runs trials in parallel across CPU cores using multiprocessing.Pool, driving
CPU utilization to 70-90% and reducing 3+ hour runtimes to ~20 minutes.
Output JSON matches the exact format of benchmarks/cells/<dataset>.json.
"""
import os
import sys
import json
import time
import platform
import argparse
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skarm.datasets import load_transactions
from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
THRESHOLDS_PATH = os.path.join(BASE, "thresholds.json")
CELLS_DIR = os.path.join(BASE, "cells")
OUT = os.path.join(BASE, "campaign_results.json")
REPORT = os.path.join(BASE, "CAMPAIGN_REPORT.md")

TRIAL_TIMEOUT_S = 600
MLXTEND_TRIALS = 5
MLXTEND_DENSE_BYTE_CAP = 2_000_000_000
ENGINES = ["apriori", "fpgrowth", "eclat"]
SUBSET_30 = ["groceries", "chess", "mushroom", "retail", "connect",
             "bms_pos", "tic_tac_toe", "docword_nips"]


def get_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def compare_itemsets(a, b, tol=1e-9):
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


# Worker function executed in process pool
def _worker_run_skarm_trial(X_npz_path, engine, thr, trial_num, return_itemsets=False):
    import scipy.sparse
    import time
    import threading
    import ctypes

    X = scipy.sparse.load_npz(X_npz_path)
    n_txns = int(X.shape[0])

    class RunTimeout(Exception):
        pass

    target_tid = threading.get_ident()
    cancel_event = threading.Event()
    def _watchdog():
        if not cancel_event.wait(TRIAL_TIMEOUT_S):
            try:
                ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [ctypes.c_ulong, ctypes.py_object]
                ctypes.pythonapi.PyThreadState_SetAsyncExc.restype = ctypes.c_int
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(target_tid),
                    ctypes.py_object(RunTimeout)
                )
            except Exception:
                pass
    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()

    baseline_rss = get_rss_mb()
    t0 = time.perf_counter()
    try:
        est = AssociationRulesEstimator(
            min_support=thr, metric="confidence", min_threshold=0.25,
            algorithm=engine, max_candidates=2_000_000, sparse_output=True,
        )
        est.fit(X)
        secs = time.perf_counter() - t0
        peak_rss = max(get_rss_mb() - baseline_rss, 0.0)
        rec = {
            "status": "ok",
            "seconds": round(secs, 4),
            "peak_mem_mb": round(peak_rss, 2),
            "n_itemsets": len(est.frequent_itemsets_),
            "n_rules": len(est.rules_),
            "is_degraded": bool(est.is_degraded_),
            "peak_candidates": int(est.pruning_policy_.peak_candidate_count),
            "threshold": thr,
            "engine": engine,
            "trial": trial_num,
            "n_transactions": n_txns,
            "throughput": round(n_txns / secs, 1) if secs > 0 else 0
        }
        itemsets = est.frequent_itemsets_ if return_itemsets else None
        return rec, itemsets
    except RunTimeout:
        secs = time.perf_counter() - t0
        return {
            "status": "timeout",
            "seconds": round(secs, 4),
            "error": "exceeded 600 s trial cap",
            "threshold": thr,
            "engine": engine,
            "trial": trial_num,
            "n_transactions": n_txns,
            "throughput": 0
        }, None
    except Exception as e:
        secs = time.perf_counter() - t0
        return {
            "status": "error",
            "seconds": round(secs, 4),
            "error": f"{type(e).__name__}: {e}",
            "threshold": thr,
            "engine": engine,
            "trial": trial_num,
            "n_transactions": n_txns,
            "throughput": 0
        }, None
    finally:
        cancel_event.set()


def _worker_run_mlxtend_trial(X_npz_path, cols_json_path, thr, trial_num, return_itemsets=False):
    import scipy.sparse
    import pandas as pd
    import time
    import threading
    import ctypes
    from mlxtend.frequent_patterns import apriori as ml_apriori
    from mlxtend.frequent_patterns import association_rules as ml_rules

    X = scipy.sparse.load_npz(X_npz_path)
    with open(cols_json_path) as f:
        columns = json.load(f)
    n_txns = int(X.shape[0])

    class RunTimeout(Exception):
        pass

    target_tid = threading.get_ident()
    cancel_event = threading.Event()
    def _watchdog():
        if not cancel_event.wait(TRIAL_TIMEOUT_S):
            try:
                ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [ctypes.c_ulong, ctypes.py_object]
                ctypes.pythonapi.PyThreadState_SetAsyncExc.restype = ctypes.c_int
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(target_tid),
                    ctypes.py_object(RunTimeout)
                )
            except Exception:
                pass
    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()

    baseline_rss = get_rss_mb()
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame(X.toarray(), columns=columns, dtype=bool)
        freq = ml_apriori(df, min_support=thr, use_colnames=True)
        rules = ml_rules(freq, metric="confidence", min_threshold=0.25)
        secs = time.perf_counter() - t0
        peak_rss = max(get_rss_mb() - baseline_rss, 0.0)
        rec = {
            "status": "ok",
            "seconds": round(secs, 4),
            "peak_mem_mb": round(peak_rss, 2),
            "n_itemsets": len(freq),
            "n_rules": len(rules),
            "threshold": thr,
            "engine": "mlxtend",
            "trial": trial_num,
            "n_transactions": n_txns,
            "throughput": round(n_txns / secs, 1) if secs > 0 else 0
        }
        named = {frozenset(r): float(s) for r, s in zip(freq["itemsets"], freq["support"])} if return_itemsets else None
        return rec, named
    except RunTimeout:
        secs = time.perf_counter() - t0
        return {
            "status": "timeout",
            "seconds": round(secs, 4),
            "error": "exceeded 600 s trial cap",
            "threshold": thr,
            "engine": "mlxtend",
            "trial": trial_num,
            "n_transactions": n_txns,
            "throughput": 0
        }, None
    except Exception as e:
        secs = time.perf_counter() - t0
        return {
            "status": "error",
            "seconds": round(secs, 4),
            "error": f"{type(e).__name__}: {e}",
            "threshold": thr,
            "engine": "mlxtend",
            "trial": trial_num,
            "n_transactions": n_txns,
            "throughput": 0
        }, None
    finally:
        cancel_event.set()


def run_dataset_parallel(name, max_workers=10):
    import scipy.sparse

    with open(THRESHOLDS_PATH) as f:
        cfg = json.load(f)["datasets"][name]

    fname = cfg["file"]
    thresholds = cfg["thresholds"]
    n_trials = 30 if name in SUBSET_30 else 5
    path = os.path.join(DATA_DIR, fname)

    print(f"\n=======================================================")
    print(f"Starting {name} ({fname}): expected {cfg['expected_n']} txns, {n_trials} trials, thresholds={thresholds}")
    print(f"Using {max_workers} parallel workers")
    print(f"=======================================================")

    t0 = time.perf_counter()
    tx = load_transactions(path)
    assert len(tx) == cfg["expected_n"], f"Transaction count mismatch: {len(tx)} != {cfg['expected_n']}"
    encoder = TransactionEncoder(sparse_output=True)
    X = encoder.fit_transform(tx)
    encode_sec = round(time.perf_counter() - t0, 3)
    columns = [str(c) for c in encoder.columns_]
    del tx
    dense_bytes = int(X.shape[0]) * int(X.shape[1])
    mlxtend_feasible = dense_bytes <= MLXTEND_DENSE_BYTE_CAP

    print(f"Encoded {name} in {encode_sec}s: shape {X.shape}, dense {dense_bytes/1e6:.1f} MB (mlxtend feasible: {mlxtend_feasible})")

    # Save X and columns to temp files for workers to load
    os.makedirs(CELLS_DIR, exist_ok=True)
    x_tmp = os.path.join(CELLS_DIR, f".tmp_{name}_X.npz")
    cols_tmp = os.path.join(CELLS_DIR, f".tmp_{name}_cols.json")
    scipy.sparse.save_npz(x_tmp, X)
    with open(cols_tmp, "w") as f:
        json.dump(columns, f)

    bundle = {
        "dataset": name,
        "records": [],
        "correctness": {},
        "skips": [],
        "encode_times": {name: encode_sec},
        "complete": False
    }

    try:
        for thr in thresholds:
            print(f"\n--- Threshold {thr} ({name}) ---")
            cell_key = f"{name}@{thr}"
            trial1_itemsets = {}

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 1. Launch trial 0 for fast engines (eclat, fpgrowth) and apriori
                print(f"Launching Trial 0 for {ENGINES} concurrently...")
                f_engs = {
                    eng: executor.submit(_worker_run_skarm_trial, x_tmp, eng, thr, 0, True)
                    for eng in ENGINES
                }

                all_remaining_futures = {}

                # Poll as futures finish to pipeline remaining trials ASAP
                pending_t0 = dict(f_engs)
                while pending_t0:
                    done_eng = None
                    for eng, fut in pending_t0.items():
                        if fut.done():
                            done_eng = eng
                            break
                    if done_eng is None:
                        time.sleep(0.5)
                        continue

                    fut = pending_t0.pop(done_eng)
                    rec, itemsets = fut.result()
                    rec["dataset"] = name
                    bundle["records"].append(rec)
                    print(f"  Trial 0 [{done_eng}]: status={rec['status']}, time={rec['seconds']}s, mem={rec.get('peak_mem_mb',0)}MB")

                    if rec["status"] == "ok" and itemsets is not None:
                        trial1_itemsets[done_eng] = itemsets
                        # Immediately pipeline trials 1..n_trials-1 for this engine!
                        print(f"  Pipelining trials 1..{n_trials-1} for {done_eng} into free workers...")
                        for t in range(1, n_trials):
                            f_rem = executor.submit(_worker_run_skarm_trial, x_tmp, done_eng, thr, t, False)
                            all_remaining_futures[f_rem] = (done_eng, t)
                    elif rec["status"] == "timeout":
                        for t in range(1, n_trials):
                            bundle["records"].append({
                                "dataset": name, "threshold": thr, "engine": done_eng, "trial": t,
                                "status": "skipped", "seconds": 0,
                                "error": "skipped after trial-0 timeout (600 s cap)"
                            })
                        bundle["skips"].append({
                            "dataset": name, "threshold": thr, "engine": done_eng,
                            "reason": "trial 0 exceeded 600 s; remaining trials skipped"
                        })
                        print(f"  --> {done_eng} timed out on trial 0; trials 1..{n_trials-1} marked skipped.")

                # Correctness verification across completed trial-0 itemsets
                verdict = {}
                engs_ok = [e for e in ENGINES if e in trial1_itemsets]
                if len(engs_ok) >= 2:
                    for i in range(len(engs_ok)):
                        for j in range(i + 1, len(engs_ok)):
                            pair_name = f"{engs_ok[i]}_vs_{engs_ok[j]}"
                            cmp_res = compare_itemsets(trial1_itemsets[engs_ok[i]], trial1_itemsets[engs_ok[j]])
                            verdict[pair_name] = cmp_res
                            print(f"  Correctness gate [{pair_name}]: match={cmp_res['match']} (itemsets={cmp_res['n_a']})")
                            assert cmp_res["match"], f"FATAL: itemset mismatch in {pair_name}: {cmp_res}"
                else:
                    verdict["note"] = f"only {len(engs_ok)} engine(s) completed trial 0; no comparison"
                    print(f"  Correctness gate note: {verdict['note']}")
                bundle["correctness"][cell_key] = verdict

                # Wait for all pipelined trials to finish
                if all_remaining_futures:
                    print(f"Waiting for {len(all_remaining_futures)} pipelined trials to complete...")
                    done_cnt = 0
                    total_cnt = len(all_remaining_futures)
                    for f in as_completed(all_remaining_futures):
                        eng, t = all_remaining_futures[f]
                        rec, _ = f.result()
                        rec["dataset"] = name
                        bundle["records"].append(rec)
                        done_cnt += 1
                        print(f"    [{done_cnt}/{total_cnt}] {eng} trial {t}: {rec['seconds']}s (status={rec['status']}, mem={rec.get('peak_mem_mb',0)}MB)")

                # Step 3: mlxtend baseline
                if not mlxtend_feasible:
                    bundle["skips"].append({
                        "dataset": name, "threshold": thr, "engine": "mlxtend",
                        "reason": f"dense boolean DataFrame would need ~{dense_bytes / 1e9:.1f} GB"
                    })
                    print(f"  mlxtend: skipped (dense dataframe needs ~{dense_bytes/1e9:.1f} GB > 2GB cap)")
                else:
                    print(f"Running mlxtend trial 0...")
                    rec0, ml_named = _worker_run_mlxtend_trial(x_tmp, cols_tmp, thr, 0, True)
                    rec0["dataset"] = name
                    bundle["records"].append(rec0)
                    print(f"  mlxtend trial 0: status={rec0['status']}, time={rec0['seconds']}s")

                    if rec0["status"] == "timeout":
                        for tt in range(1, MLXTEND_TRIALS):
                            bundle["records"].append({
                                "dataset": name, "threshold": thr, "engine": "mlxtend",
                                "trial": tt, "status": "skipped", "seconds": 0,
                                "error": "skipped after timeout (600 s cap)"
                            })
                        bundle["skips"].append({
                            "dataset": name, "threshold": thr, "engine": "mlxtend",
                            "reason": "trial 0 exceeded 600 s; remaining trials skipped"
                        })
                    elif rec0["status"] == "ok":
                        if ml_named is not None and "apriori" in trial1_itemsets:
                            named_skarm = {frozenset(columns[i] for i in k): v
                                           for k, v in trial1_itemsets["apriori"].items()}
                            bundle["correctness"][cell_key]["skarm_apriori_vs_mlxtend"] = \
                                compare_itemsets(named_skarm, ml_named)
                            print(f"  mlxtend vs skarm_apriori match: {bundle['correctness'][cell_key]['skarm_apriori_vs_mlxtend']['match']}")

                        print(f"Running mlxtend trials 1..{MLXTEND_TRIALS-1} in parallel...")
                        ml_futs = {
                            executor.submit(_worker_run_mlxtend_trial, x_tmp, cols_tmp, thr, t, False): t
                            for t in range(1, MLXTEND_TRIALS)
                        }
                        for f in as_completed(ml_futs):
                            t = ml_futs[f]
                            rec, _ = f.result()
                            rec["dataset"] = name
                            bundle["records"].append(rec)
                            print(f"  mlxtend trial {t}: {rec['seconds']}s (status={rec['status']})")

        bundle["complete"] = True
        cell_file = os.path.join(CELLS_DIR, f"{name}.json")
        cell_tmp = os.path.join(CELLS_DIR, f".{name}.json.tmp")
        with open(cell_tmp, "w") as f:
            json.dump(bundle, f, indent=2)
        os.replace(cell_tmp, cell_file)
        print(f"\nSUCCESS: Saved {cell_file} with {len(bundle['records'])} records ({sum(1 for r in bundle['records'] if r['status']=='ok')} ok)")

    finally:
        # Clean up temp files
        for p in [x_tmp, cols_tmp]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset key (e.g. t10i4d100k or bms_pos)")
    parser.add_argument("--workers", type=int, default=10, help="number of worker processes")
    args = parser.parse_args()
    run_dataset_parallel(args.dataset, max_workers=args.workers)
