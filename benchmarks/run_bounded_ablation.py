"""Automated Bounded Ablation Sweep Harness for SKARM.

Quantifies the trade-offs between candidate budget caps (and length limits)
versus pattern recall, precision, and execution speedup across representative
benchmark datasets:
  - Dense: chess @ min_support=0.85
  - Medium: adult_census @ min_support=0.4
  - Sparse: bms_pos @ min_support=0.05

Outputs machine-readable metrics to benchmarks/ablation_results.json.
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skarm.datasets import load_transactions
from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator

DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ablation_results.json")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SWEEP_CONFIGS = {
    "chess": {
        "file": "chess.dat",
        "support": 0.85,
        "type": "dense",
        "description": "Dense game board (N=3,196, 49.3% density, L_avg=37.0)",
        "algorithm": "apriori",
    },
    "adult_census": {
        "file": "adult_census.dat",
        "support": 0.40,
        "type": "medium",
        "description": "Medium census survey (N=32,561, 6.2% density, L_avg=13.9)",
        "algorithm": "fpgrowth",
    },
    "bms_pos": {
        "file": "BMS-POS.csv",
        "support": 0.05,
        "type": "sparse",
        "description": "Sparse point-of-sale clickstream (N=515,596, 0.4% density, L_avg=6.5)",
        "algorithm": "eclat",
    },
}

DEFAULT_CANDIDATE_CAPS = [50, 100, 250, 500, 1000, 2500, 5000]
DEFAULT_MAX_LENS = [1, 2, 3, 4, 5, None]


def extract_rule_tuples(rules_df) -> Set[Tuple[frozenset, frozenset]]:
    """Convert rules DataFrame into set of (antecedent_set, consequent_set) tuples."""
    if rules_df is None or rules_df.empty:
        return set()
    return {
        (frozenset(row["antecedent"]), frozenset(row["consequent"]))
        for _, row in rules_df.iterrows()
    }


def run_ablation_sweep(
    datasets: List[str],
    candidate_caps: List[int],
    max_lens: List[Any],
    data_dir: str = DEFAULT_DATA_DIR,
    output_path: str = DEFAULT_OUTPUT_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute automated candidate and length sweeps against unconstrained baselines."""
    import subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit = "ea5bff5d657639298b197074ab35bd4bb7bb7a37"

    import numpy as np
    import pandas as pd
    import sklearn
    import scipy

    results: Dict[str, Any] = {
        "metadata": {
            "title": "SKARM Bounded vs Exact Pruning Ablation Campaign",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit,
            "platform": f"{platform.machine()}, {platform.system()} {platform.release()}",
            "python_version": sys.version.split()[0],
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "scipy": scipy.__version__,
            },
            "candidate_caps": candidate_caps,
            "max_lens": [str(m) if m is not None else "None" for m in max_lens],
            "datasets": {},
        },
        "candidate_sweeps": {},
        "length_sweeps": {},
    }

    for name in datasets:
        if name not in SWEEP_CONFIGS:
            print(f"[WARN] Unknown dataset {name}, skipping.")
            continue

        cfg = SWEEP_CONFIGS[name]
        results["metadata"]["datasets"][name] = cfg
        file_path = os.path.join(data_dir, cfg["file"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file {file_path} not found.")

        if verbose:
            print(f"\n=======================================================")
            print(f"Loading {name} ({cfg['description']})...")

        t0_load = time.perf_counter()
        raw_tx = load_transactions(file_path)
        encoder = TransactionEncoder(sparse_output=True)
        X = encoder.fit_transform(raw_tx)
        t_load = time.perf_counter() - t0_load

        if verbose:
            print(f"Loaded {X.shape[0]} transactions, {X.shape[1]} items in {t_load:.2f}s.")
            print(f"Running EXACT baseline at min_support={cfg['support']} with {cfg['algorithm']}...")

        # 1. Compute EXACT ground truth baseline (uncapped candidates, uncapped length)
        t0_exact = time.perf_counter()
        est_exact = AssociationRulesEstimator(
            min_support=cfg["support"],
            metric="confidence",
            min_threshold=0.25,
            algorithm=cfg["algorithm"],
            max_candidates=10_000_000,
            on_candidate_explosion="raise",
            sparse_output=True,
        )
        est_exact.fit(X)
        exact_fit_time = time.perf_counter() - t0_exact

        exact_itemsets = set(est_exact.frequent_itemsets_.keys())
        exact_rules = extract_rule_tuples(est_exact.rules_)
        n_exact_itemsets = len(exact_itemsets)
        n_exact_rules = len(exact_rules)

        if verbose:
            print(f"EXACT baseline: {n_exact_itemsets} itemsets, {n_exact_rules} rules in {exact_fit_time:.4f}s "
                  f"(status={est_exact.status_}, degraded={est_exact.is_degraded_})")

        baseline_summary = {
            "n_itemsets": n_exact_itemsets,
            "n_rules": n_exact_rules,
            "fit_time_s": round(exact_fit_time, 4),
            "status": est_exact.status_,
            "is_degraded": est_exact.is_degraded_,
            "counters": est_exact.metadata_.get("counters", {}),
        }

        # 2. Candidate Cap Sweep
        candidate_records = []
        if verbose:
            print(f"\n--- Sweeping Candidate Caps {candidate_caps} ---")

        for cap in candidate_caps:
            t0_cap = time.perf_counter()
            est_cap = AssociationRulesEstimator(
                min_support=cfg["support"],
                metric="confidence",
                min_threshold=0.25,
                algorithm=cfg["algorithm"],
                max_candidates=cap,
                on_candidate_explosion="degrade",
                sparse_output=True,
            )
            est_cap.fit(X)
            cap_fit_time = time.perf_counter() - t0_cap

            cap_itemsets = set(est_cap.frequent_itemsets_.keys())
            cap_rules = extract_rule_tuples(est_cap.rules_)

            # Recall and precision calculations
            itemset_recall = len(cap_itemsets & exact_itemsets) / n_exact_itemsets if n_exact_itemsets > 0 else 1.0
            itemset_precision = len(cap_itemsets & exact_itemsets) / len(cap_itemsets) if cap_itemsets else 1.0
            rule_recall = len(cap_rules & exact_rules) / n_exact_rules if n_exact_rules > 0 else 1.0
            rule_precision = len(cap_rules & exact_rules) / len(cap_rules) if cap_rules else 1.0
            speedup = exact_fit_time / cap_fit_time if cap_fit_time > 0 else 0.0

            rec = {
                "candidate_cap": cap,
                "n_itemsets": len(cap_itemsets),
                "n_rules": len(cap_rules),
                "itemset_recall": round(itemset_recall, 6),
                "itemset_precision": round(itemset_precision, 6),
                "rule_recall": round(rule_recall, 6),
                "rule_precision": round(rule_precision, 6),
                "fit_time_s": round(cap_fit_time, 4),
                "speedup_factor": round(speedup, 2),
                "status_": est_cap.status_,
                "is_degraded_": est_cap.is_degraded_,
                "future_candidates_status": est_cap.metadata_.get("future_candidates_status"),
                "termination_reason_": est_cap.termination_reason_,
                "level_halted": est_cap.metadata_.get("level_halted"),
                "candidates_generated_at_level": est_cap.metadata_.get("candidates_generated_at_level"),
                "candidates_remaining_at_level": est_cap.metadata_.get("candidates_remaining_at_level"),
                "peak_candidates": int(est_cap.pruning_policy_.peak_candidate_count),
                "counters": est_cap.metadata_.get("counters", {}),
            }
            candidate_records.append(rec)
            if verbose:
                print(f"  Cap={cap:5d}: itemsets={len(cap_itemsets):5d} (recall={itemset_recall*100:6.2f}%), "
                      f"rules={len(cap_rules):5d} (recall={rule_recall*100:6.2f}%), "
                      f"time={cap_fit_time:7.4f}s (speedup={speedup:6.1f}x), "
                      f"status={est_cap.status_}")

        results["candidate_sweeps"][name] = {
            "baseline": baseline_summary,
            "records": candidate_records,
        }

        # 3. Length Limit Sweep
        length_records = []
        if verbose:
            print(f"\n--- Sweeping Max Length Limits {max_lens} ---")

        for m_len in max_lens:
            t0_len = time.perf_counter()
            est_len = AssociationRulesEstimator(
                min_support=cfg["support"],
                metric="confidence",
                min_threshold=0.25,
                algorithm=cfg["algorithm"],
                max_len=m_len,
                max_candidates=10_000_000,
                on_candidate_explosion="degrade",
                sparse_output=True,
            )
            est_len.fit(X)
            len_fit_time = time.perf_counter() - t0_len

            len_itemsets = set(est_len.frequent_itemsets_.keys())
            len_rules = extract_rule_tuples(est_len.rules_)

            itemset_recall = len(len_itemsets & exact_itemsets) / n_exact_itemsets if n_exact_itemsets > 0 else 1.0
            itemset_precision = len(len_itemsets & exact_itemsets) / len(len_itemsets) if len_itemsets else 1.0
            rule_recall = len(len_rules & exact_rules) / n_exact_rules if n_exact_rules > 0 else 1.0
            rule_precision = len(len_rules & exact_rules) / len(len_rules) if len_rules else 1.0
            speedup = exact_fit_time / len_fit_time if len_fit_time > 0 else 0.0

            l_label = str(m_len) if m_len is not None else "None (Exact)"
            l_rec = {
                "max_len": m_len,
                "label": l_label,
                "n_itemsets": len(len_itemsets),
                "n_rules": len(len_rules),
                "itemset_recall": round(itemset_recall, 6),
                "itemset_precision": round(itemset_precision, 6),
                "rule_recall": round(rule_recall, 6),
                "rule_precision": round(rule_precision, 6),
                "fit_time_s": round(len_fit_time, 4),
                "speedup_factor": round(speedup, 2),
                "status_": est_len.status_,
                "is_degraded_": est_len.is_degraded_,
                "future_candidates_status": est_len.metadata_.get("future_candidates_status"),
                "counters": est_len.metadata_.get("counters", {}),
            }
            length_records.append(l_rec)
            if verbose:
                print(f"  Length={l_label:12s}: itemsets={len(len_itemsets):5d} (recall={itemset_recall*100:6.2f}%), "
                      f"rules={len(len_rules):5d} (recall={rule_recall*100:6.2f}%), "
                      f"time={len_fit_time:7.4f}s (speedup={speedup:6.1f}x), "
                      f"status={est_len.status_}")

        results["length_sweeps"][name] = {
            "baseline": baseline_summary,
            "records": length_records,
        }

    # Write output to json
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n[DONE] Bounded ablation results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SKARM Bounded Ablation Sweep Harness")
    parser.add_argument(
        "--datasets",
        default="chess,adult_census,bms_pos",
        help="Comma-separated dataset keys (default: chess,adult_census,bms_pos)",
    )
    parser.add_argument(
        "--caps",
        default="50,100,250,500,1000,2500,5000",
        help="Comma-separated candidate caps (default: 50,100,250,500,1000,2500,5000)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset files directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to output results JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    caps = [int(c.strip()) for c in args.caps.split(",") if c.strip()]

    run_ablation_sweep(
        datasets=datasets,
        candidate_caps=caps,
        max_lens=DEFAULT_MAX_LENS,
        data_dir=args.data_dir,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    main()
