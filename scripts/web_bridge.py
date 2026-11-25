"""Safe stdin/stdout JSON bridge invoking skarm for the Node.js workbench server."""
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from skarm.datasets import (
    dataset_summary,
    load_fimi_dat,
    load_groceries,
    load_market_basket,
    load_mushroom,
)
from skarm.estimator import AssociationRulesEstimator

DEMO_TRANSACTIONS = [
    ["milk", "bread", "butter"],
    ["bread", "butter"],
    ["milk", "bread"],
    ["beer", "diapers"],
    ["milk", "beer", "diapers", "bread"],
    ["bread", "butter", "milk"],
    ["diapers", "beer"],
    ["milk", "bread", "butter", "eggs"],
    ["milk", "bread", "eggs"],
    ["beer", "diapers", "chips"],
]


def _safe_float(val):
    if val is None:
        return None
    f = float(val)
    if math.isnan(f):
        return None
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return round(f, 6)


def _load_dataset(name, custom_txns):
    if name == "demo":
        return "Synthetic Market Basket Demo", DEMO_TRANSACTIONS
    if name == "groceries":
        return "Groceries Market Basket (9,835 txns)", load_groceries(
            os.path.join(REPO_ROOT, "data", "groceries.csv")
        )
    if name == "market_basket":
        return "Market Basket Optimisation (7,501 txns)", load_market_basket(
            os.path.join(REPO_ROOT, "data", "market_basket.csv")
        )
    if name == "tic_tac_toe":
        return "UCI Tic-Tac-Toe Endgame (958 txns)", load_fimi_dat(
            os.path.join(REPO_ROOT, "data", "tic_tac_toe.dat")
        )
    if name == "mushroom":
        return "UCI Mushroom (8,124 txns)", load_mushroom(
            os.path.join(REPO_ROOT, "data", "mushroom.dat")
        )
    if name == "custom":
        return "Custom User Transactions", custom_txns
    raise ValueError(f"Unsupported dataset: {name}")


def main():
    raw = sys.stdin.read()
    payload = json.loads(raw)

    dataset = payload["dataset"]
    algorithm = payload["algorithm"]
    metric = payload["metric"]
    min_support = float(payload["min_support"])
    min_threshold = float(payload["min_threshold"])
    max_candidates = int(payload["max_candidates"])
    top_n = int(payload.get("top_n", 50))
    custom_txns = payload.get("transactions")
    basket = payload.get("basket") or []

    dataset_label, transactions = _load_dataset(dataset, custom_txns)
    summary = dataset_summary(transactions)

    est = AssociationRulesEstimator(
        min_support=min_support,
        metric=metric,
        min_threshold=min_threshold,
        algorithm=algorithm,
        max_candidates=max_candidates,
        sparse_output=True,
    )

    t0 = time.perf_counter()
    est.fit(transactions)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    rules_df = est.get_rules()
    total_rules = len(rules_df)
    total_itemsets = len(est.frequent_itemsets_)

    # Build itemset support lookup for rule formula explanation
    itemset_supp = {}
    if isinstance(est.frequent_itemsets_, dict):
        for fs, sp in est.frequent_itemsets_.items():
            itemset_supp[frozenset(str(feature_names_val) for feature_names_val in (
                (getattr(est, "feature_names_in_", [])[i] if isinstance(i, int) else i) for i in fs
            ))] = float(sp)
    elif total_itemsets > 0:
        for _, r in est.frequent_itemsets_.iterrows():
            names = r["itemset_names"] if "itemset_names" in r else r["itemset"]
            itemset_supp[frozenset(str(x) for x in names)] = float(r["support"])

    if total_rules > 0 and metric in rules_df.columns:
        rules_df = rules_df.sort_values(
            by=[metric, "confidence", "support"], ascending=False
        )

    top_rules = []
    for _, row in rules_df.head(top_n).iterrows():
        ant = row["antecedent_names"] if "antecedent_names" in row else row["antecedent"]
        con = row["consequent_names"] if "consequent_names" in row else row["consequent"]
        ant_set = frozenset(str(x) for x in ant)
        con_set = frozenset(str(x) for x in con)
        top_rules.append(
            {
                "antecedent": sorted(ant_set),
                "consequent": sorted(con_set),
                "antecedent_support": _safe_float(itemset_supp.get(ant_set)),
                "consequent_support": _safe_float(itemset_supp.get(con_set)),
                "support": _safe_float(row.get("support")),
                "confidence": _safe_float(row.get("confidence")),
                "lift": _safe_float(row.get("lift")),
                "leverage": _safe_float(row.get("leverage")),
                "conviction": _safe_float(row.get("conviction")),
                "zhangs_metric": _safe_float(row.get("zhangs_metric")),
            }
        )

    feature_names = list(getattr(est, "feature_names_in_", []))

    top_itemsets = []
    if total_itemsets > 0:
        if isinstance(est.frequent_itemsets_, dict):
            sorted_items = sorted(
                est.frequent_itemsets_.items(),
                key=lambda kv: (-float(kv[1]), len(kv[0]), sorted(kv[0])),
            )[:20]
            for itemset_idx, supp in sorted_items:
                names = [
                    str(feature_names[i]) if isinstance(i, int) and 0 <= i < len(feature_names) else str(i)
                    for i in itemset_idx
                ]
                top_itemsets.append(
                    {
                        "items": sorted(names),
                        "length": len(names),
                        "support": _safe_float(supp),
                    }
                )
        else:
            itemsets_sorted = est.frequent_itemsets_.sort_values(
                by=["support"], ascending=False
            ).head(20)
            for _, row in itemsets_sorted.iterrows():
                names = row["itemset_names"] if "itemset_names" in row else row["itemset"]
                top_itemsets.append(
                    {
                        "items": sorted(str(x) for x in names),
                        "length": len(names),
                        "support": _safe_float(row.get("support")),
                    }
                )

    recommendations = []
    if basket and total_rules > 0:
        basket_set = {str(x).strip().lower() for x in basket if str(x).strip()}
        matched = []
        for _, row in rules_df.iterrows():
            ant_raw = row["antecedent_names"] if "antecedent_names" in row else row["antecedent"]
            con_raw = row["consequent_names"] if "consequent_names" in row else row["consequent"]
            ant_list = sorted(str(x) for x in ant_raw)
            con_list = sorted(str(x) for x in con_raw)
            ant_lower = {x.lower() for x in ant_list}
            con_lower = {x.lower() for x in con_list}
            if ant_lower.issubset(basket_set) and not con_lower.issubset(basket_set):
                matched.append(
                    {
                        "matched_antecedent": ant_list,
                        "recommended": [x for x in con_list if x.lower() not in basket_set],
                        "confidence": _safe_float(row.get("confidence")),
                        "lift": _safe_float(row.get("lift")),
                        "support": _safe_float(row.get("support")),
                    }
                )
        matched.sort(
            key=lambda r: (
                -(r["confidence"] or 0.0),
                -(r["lift"] or 0.0),
                -(r["support"] or 0.0),
            )
        )
        recommendations = matched[:8]

    sample_items = sorted(str(x) for x in feature_names)[:30]
    density_pct = round(
        (summary["avg_transaction_length"] / max(1, summary["n_unique_items"])) * 100.0,
        2,
    )
    meta = getattr(est, "metadata_", {}) or {}

    engine_comparison = None
    if payload.get("compare_engines"):
        engine_comparison = []
        ref_itemsets = None
        for eng in ("apriori", "fpgrowth", "eclat"):
            eng_est = AssociationRulesEstimator(
                min_support=min_support,
                metric=metric,
                min_threshold=min_threshold,
                algorithm=eng,
                max_candidates=max_candidates,
                sparse_output=True,
            )
            t_s = time.perf_counter()
            eng_est.fit(transactions)
            eng_ms = (time.perf_counter() - t_s) * 1000.0
            eng_rules = eng_est.get_rules()
            if isinstance(eng_est.frequent_itemsets_, dict):
                eng_map = {frozenset(k): float(v) for k, v in eng_est.frequent_itemsets_.items()}
            else:
                eng_map = {
                    frozenset(r["itemset"]): float(r["support"])
                    for _, r in eng_est.frequent_itemsets_.iterrows()
                }
            if ref_itemsets is None:
                ref_itemsets = eng_map
                delta_supp = 0.0
                exact_match = True
            else:
                keys_match = set(ref_itemsets.keys()) == set(eng_map.keys())
                delta_supp = (
                    max((abs(ref_itemsets[k] - eng_map[k]) for k in ref_itemsets), default=0.0)
                    if keys_match
                    else 1.0
                )
                exact_match = bool(keys_match and delta_supp <= 1e-9)
            engine_comparison.append(
                {
                    "algorithm": eng,
                    "elapsed_ms": round(eng_ms, 3),
                    "n_itemsets": len(eng_map),
                    "n_rules": len(eng_rules),
                    "status": getattr(eng_est, "status_", "EXACT"),
                    "exact_parity": exact_match,
                    "max_delta_supp": _safe_float(delta_supp),
                }
            )

    response = {
        "dataset": dataset,
        "dataset_label": dataset_label,
        "summary": {**summary, "density_pct": density_pct},
        "algorithm_requested": algorithm,
        "algorithm_used": getattr(est, "algorithm_used_", algorithm),
        "status": getattr(est, "status_", "EXACT"),
        "is_degraded": bool(getattr(est, "is_degraded_", False)),
        "termination_reason": getattr(est, "termination_reason_", None),
        "peak_candidates": meta.get("peak_candidates"),
        "elapsed_ms": round(elapsed_ms, 3),
        "n_itemsets": total_itemsets,
        "n_rules": total_rules,
        "rules": top_rules,
        "top_itemsets": top_itemsets,
        "recommendations": recommendations,
        "sample_items": sample_items,
        "engine_comparison": engine_comparison,
    }
    sys.stdout.write(json.dumps(response))


if __name__ == "__main__":
    main()
