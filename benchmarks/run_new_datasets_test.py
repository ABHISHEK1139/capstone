"""Empirical evaluation and real testing on newly downloaded datasets:
1. Market Basket Optimisation (N=7,501, French retail point-of-sale)
2. Mushroom Benchmark (N=8,124, classic dense FIMI benchmark)
"""
import sys
import os
import time
import json
import psutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skarm.datasets import load_market_basket, load_mushroom, dataset_summary
from skarm.estimator import AssociationRulesEstimator
from skarm.explorer import RuleExplorer

def get_peak_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_market_basket():
    print("=" * 75)
    print("TEST 1: MARKET BASKET OPTIMISATION DATASET (N=7,501)")
    print("=" * 75)
    tx = load_market_basket("data/market_basket.csv")
    summary = dataset_summary(tx)
    print(f"Stats: {summary['n_transactions']:,} txns, {summary['n_unique_items']} items, avg len {summary['avg_transaction_length']}, density {summary['density_pct']}%\n")

    results = {}
    min_sup = 0.03
    min_conf = 0.20

    for algo in ["fpgrowth", "eclat", "apriori"]:
        print(f"Executing {algo.upper()} (min_sup={min_sup}, min_conf={min_conf})...")
        m0 = get_peak_memory_mb()
        t0 = time.perf_counter()

        est = AssociationRulesEstimator(
            min_support=min_sup,
            metric="confidence",
            min_threshold=min_conf,
            algorithm=algo,
            sparse_output=True,
        )
        est.fit(tx)
        elapsed = time.perf_counter() - t0
        m1 = get_peak_memory_mb()
        peak_ram = max(m1 - m0, 0.1)

        n_itemsets = len(est.frequent_itemsets_)
        n_rules = len(est.rules_)
        tput = len(tx) / elapsed if elapsed > 0 else 0
        print(f"  -> {algo.upper()}: {elapsed:.3f}s | Peak RAM: {peak_ram:.1f} MB | Itemsets: {n_itemsets} | Rules: {n_rules} | {tput:.1f} tx/s")

        results[algo] = {
            "elapsed_sec": round(elapsed, 4),
            "peak_ram_mb": round(peak_ram, 2),
            "n_itemsets": n_itemsets,
            "n_rules": n_rules,
            "throughput_tx_per_sec": round(tput, 1),
        }

    # Extract actionable rules from FP-Growth
    est_rules = AssociationRulesEstimator(
        min_support=0.02,
        metric="lift",
        min_threshold=1.5,
        algorithm="fpgrowth",
        sparse_output=True,
    ).fit(tx)
    explorer = est_rules.explore()
    top_rules = explorer.filter_rules(min_lift=1.5, sort_by="lift", ascending=False)
    print("\nTop Discovered Market Basket Rules (Ranked by Lift):")
    print(explorer.to_markdown(top_rules.head(10)))

    # Convert sets to list for clean JSON
    rule_records = []
    for r in top_rules.head(10).to_dict(orient="records"):
        record = {}
        for k, v in r.items():
            if isinstance(v, (set, frozenset)):
                record[k] = sorted(list(v))
            else:
                record[k] = v
        rule_records.append(record)

    return {"summary": summary, "benchmarks": results, "top_rules": rule_records}

def test_mushroom():
    print("\n" + "=" * 75)
    print("TEST 2: MUSHROOM BENCHMARK DATASET (N=8,124, 119 ATTRIBUTES)")
    print("=" * 75)
    tx = load_mushroom("data/mushroom.dat")
    summary = dataset_summary(tx)
    print(f"Stats: {summary['n_transactions']:,} txns, {summary['n_unique_items']} items, avg len {summary['avg_transaction_length']}, density {summary['density_pct']}%\n")

    results = {}
    min_sup = 0.40
    min_conf = 0.80

    for algo in ["fpgrowth", "eclat", "apriori"]:
        print(f"Executing {algo.upper()} (min_sup={min_sup}, min_conf={min_conf})...")
        m0 = get_peak_memory_mb()
        t0 = time.perf_counter()

        est = AssociationRulesEstimator(
            min_support=min_sup,
            metric="confidence",
            min_threshold=min_conf,
            algorithm=algo,
            sparse_output=True,
        )
        est.fit(tx)
        elapsed = time.perf_counter() - t0
        m1 = get_peak_memory_mb()
        peak_ram = max(m1 - m0, 0.1)

        n_itemsets = len(est.frequent_itemsets_)
        n_rules = len(est.rules_)
        tput = len(tx) / elapsed if elapsed > 0 else 0
        print(f"  -> {algo.upper()}: {elapsed:.3f}s | Peak RAM: {peak_ram:.1f} MB | Itemsets: {n_itemsets} | Rules: {n_rules} | {tput:.1f} tx/s")

        results[algo] = {
            "elapsed_sec": round(elapsed, 4),
            "peak_ram_mb": round(peak_ram, 2),
            "n_itemsets": n_itemsets,
            "n_rules": n_rules,
            "throughput_tx_per_sec": round(tput, 1),
        }

    return {"summary": summary, "benchmarks": results}

def main():
    res_market = test_market_basket()
    res_mushroom = test_mushroom()

    all_res = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Intel/AMD x86_64, Windows",
        "market_basket": res_market,
        "mushroom": res_mushroom,
    }

    def json_serializer(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (set, frozenset)):
            return sorted(list(o))
        return str(o)

    out_file = "benchmarks/new_datasets_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, default=json_serializer)
    print(f"\n[SUCCESS] New real dataset testing complete! Results saved to {out_file}")

if __name__ == "__main__":
    main()
