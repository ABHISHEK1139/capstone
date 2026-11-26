"""Safe stdin/stdout JSON bridge invoking skarm for the Node.js workbench server."""
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from skarm.datasets import dataset_summary, load_transactions
from skarm.estimator import AssociationRulesEstimator

DATASET_CATALOG = {
    "groceries": ("Groceries Market Basket (9,835 txns)", "data/groceries.csv"),
    "market_basket": ("Market Basket Optimisation (7,501 txns)", "data/market_basket.csv"),
    "retail": ("Belgian Retail Store (88,162 txns)", "data/retail.dat"),
    "online_retail": ("Online Retail UK (20,728 txns)", "data/online_retail.dat"),
    "online_retail_ii": ("Online Retail II (22,103 txns)", "data/online_retail_ii.dat"),
    "bms_webview_1": ("BMS-WebView-1 Clickstream (59,009 txns)", "data/Transactional_BMS-WebView-1.csv"),
    "bms_webview_2": ("BMS-WebView-2 Clickstream (77,512 txns)", "data/BMS-WebView-2.dat"),
    "t10i4d100k": ("IBM Quest T10I4D100K (100,000 txns)", "data/T10I4D100K.dat"),
    "tic_tac_toe": ("UCI Tic-Tac-Toe Endgame (958 txns)", "data/tic_tac_toe.dat"),
    "car_evaluation": ("UCI Car Evaluation (1,728 txns)", "data/car_evaluation.dat"),
    "balance_scale": ("UCI Balance Scale (625 txns)", "data/balance_scale.dat"),
    "mushroom": ("UCI Mushroom (8,124 txns · Dense)", "data/mushroom.dat"),
    "chess": ("UCI Chess King-Rook (3,196 txns · Dense)", "data/chess.dat"),
    "adult_census": ("UCI Adult Census (32,561 txns)", "data/adult_census.dat"),
    "docword_nips": ("UCI Docword NIPS (1,500 docs · Dense Text)", "data/docword_nips.dat"),
    "pumsb_star": ("Pumsb-Star Census (49,046 txns)", "data/pumsb_star.dat"),
    "synthetic_sparse_2000": ("Synthetic Sparse Control (2,000 txns)", "data/synthetic_sparse_2000.dat"),
    "synthetic_dense_5000": ("Synthetic Dense Control (5,000 txns)", "data/synthetic_dense_5000.dat"),
}

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
        return "Classroom Toy Matrix (10 txns)", DEMO_TRANSACTIONS
    if name == "custom":
        return "Custom User Transactions", custom_txns
    entry = DATASET_CATALOG.get(name)
    if entry is not None:
        label, rel_path = entry
        return label, load_transactions(os.path.join(REPO_ROOT, rel_path))
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
    max_len_raw = payload.get("max_len")
    max_len = int(max_len_raw) if max_len_raw is not None else None
    top_n = int(payload.get("top_n", 50))
    custom_txns = payload.get("transactions")
    basket = payload.get("basket") or []
    vip_item = (payload.get("vip_item") or "").strip() or None
    vip_multiplier = float(payload.get("vip_multiplier", 1.0))

    dataset_label, transactions = _load_dataset(dataset, custom_txns)
    summary = dataset_summary(transactions)

    sample_weights = None
    weighted_info = None
    if vip_item and vip_multiplier > 1.0:
        vip_lower = vip_item.lower()
        weights_list = [
            vip_multiplier if any(str(tok).strip().lower() == vip_lower for tok in tx) else 1.0
            for tx in transactions
        ]
        sample_weights = np.asarray(weights_list, dtype=float)
        affected_count = sum(1 for w in weights_list if w > 1.0)
        weighted_info = {
            "active": True,
            "vip_item": vip_item,
            "multiplier": vip_multiplier,
            "affected_txns": affected_count,
            "total_weight": round(float(sample_weights.sum()), 2),
        }

    est = AssociationRulesEstimator(
        min_support=min_support,
        metric=metric,
        min_threshold=min_threshold,
        algorithm=algorithm,
        max_len=max_len,
        max_candidates=max_candidates,
        sparse_output=True,
    )

    t0 = time.perf_counter()
    est.fit(transactions, sample_weight=sample_weights)
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
        for _, row in rules_df.head(min(total_rules, 2500)).iterrows():
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
    operations = meta.get("operations") or meta.get("counters") or {}
    n_rows = int(summary["n_transactions"])
    n_cols = max(1, int(summary["n_unique_items"]))
    nnz_est = int(round(float(getattr(est, "density_", 0.0)) * n_rows * n_cols))
    csr_bytes = (n_rows + 1) * 4 + nnz_est * 5
    dense_float64_bytes = n_rows * n_cols * 8
    memory_profile = {
        "csr_kb": round(csr_bytes / 1024.0, 2),
        "dense_float64_kb": round(dense_float64_bytes / 1024.0, 2),
        "nnz": nnz_est,
        "saving_pct": round((1.0 - csr_bytes / max(1, dense_float64_bytes)) * 100.0, 2),
    }

    # Live scikit-learn Pipeline TransformerMixin inspect: transform(), get_feature_names_out(), score()
    probe_basket = [str(x).strip() for x in basket if str(x).strip()]
    if not probe_basket and transactions:
        probe_basket = [str(x) for x in transactions[0][:2]]
    pipeline_transform = None
    estimator_score = None
    if total_rules > 0 and probe_basket:
        try:
            t_vec = est.transform([probe_basket])[0]
            fired_idx = [idx for idx, flag in enumerate(t_vec) if bool(flag)]
            if total_rules <= 2500:
                f_names = list(est.get_feature_names_out())
                fired_rules = [str(f_names[idx]) for idx in fired_idx[:8]]
                total_feats = len(f_names)
            else:
                total_feats = total_rules
                fired_rules = []
                for idx in fired_idx[:8]:
                    r_row = est.rules_.iloc[idx]
                    ants = sorted(str(feature_names[i]) if i < len(feature_names) else str(i) for i in r_row["antecedent"])
                    cons = sorted(str(feature_names[i]) if i < len(feature_names) else str(i) for i in r_row["consequent"])
                    fired_rules.append(f"{{{', '.join(ants)}}} -> {{{', '.join(cons)}}}")
            pipeline_transform = {
                "query_items": probe_basket,
                "fired_count": len(fired_idx),
                "total_features": total_feats,
                "active_rate_pct": round((len(fired_idx) / max(1, total_feats)) * 100.0, 1),
                "vector_preview": [int(bool(b)) for b in t_vec[:24]],
                "fired_rules": fired_rules,
            }
            estimator_score = _safe_float(
                est.score(transactions) if (total_rules * n_rows <= 250000) else est.score()
            )
        except Exception:
            pipeline_transform = None

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
                max_len=max_len,
                max_candidates=max_candidates,
                sparse_output=True,
            )
            t_s = time.perf_counter()
            eng_est.fit(transactions, sample_weight=sample_weights)
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

    algo_used = getattr(est, "algorithm_selected_", getattr(est, "algorithm_used_", algorithm))
    response = {
        "dataset": dataset,
        "dataset_label": dataset_label,
        "summary": {**summary, "density_pct": density_pct},
        "algorithm_requested": algorithm,
        "algorithm_used": algo_used,
        "status": getattr(est, "status_", "EXACT"),
        "is_degraded": bool(getattr(est, "is_degraded_", False)),
        "termination_reason": getattr(est, "termination_reason_", None),
        "peak_candidates": meta.get("candidates_observed_before_halt") or operations.get("candidates_generated"),
        "level_halted": meta.get("level_halted"),
        "future_candidates_status": meta.get("future_candidates_status"),
        "operations": operations,
        "memory_profile": memory_profile,
        "weighted_info": weighted_info,
        "pipeline_transform": pipeline_transform,
        "estimator_score": estimator_score,
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
