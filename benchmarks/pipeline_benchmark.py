"""Pipeline & Scikit-Learn Integration Benchmark for SKARM.

Verifies and benchmarks:
1. End-to-end Pipeline([('encoder', TransactionEncoder()), ('rules', AssociationRulesEstimator())])
   across dense and sparse representations.
2. Scikit-learn .set_output(transform="pandas") compatibility and feature naming.
3. GridSearchCV hyperparameter tuning across min_support and min_threshold using
   unsupervised metrics (mean_confidence_scorer, total_support_scorer).
4. Downstream transactional classification: Pipeline([('encoder', ...), ('rules', ...), ('clf', LogisticRegression())])
   evaluating accuracy, precision, recall, and F1 on real datasets (tic_tac_toe and mushroom).

Saves machine-readable results to benchmarks/pipeline_results.json.
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skarm.datasets import load_transactions
from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator
from skarm.metrics import mean_confidence_scorer, total_support_scorer, mean_lift_scorer

DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "pipeline_results.json")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def benchmark_pipeline_basics(data_dir: str) -> Dict[str, Any]:
    """Benchmark end-to-end Pipeline with TransactionEncoder and AssociationRulesEstimator."""
    print("\n--- Part 1: End-to-End Pipeline([('encoder', ...), ('rules', ...)]) ---")

    # Load groceries dataset as representative transactional data
    groceries_path = os.path.join(data_dir, "groceries.csv")
    transactions = load_transactions(groceries_path)

    results: Dict[str, Any] = {
        "dataset": "groceries.csv",
        "n_transactions": len(transactions),
        "sparse_mode": {},
        "dense_mode": {},
    }

    # 1.1 Sparse Mode Pipeline
    t0 = time.perf_counter()
    pipe_sparse = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=True)),
        ("rules", AssociationRulesEstimator(
            min_support=0.02,
            metric="confidence",
            min_threshold=0.25,
            algorithm="auto",
            sparse_output=True,
        )),
    ])
    pipe_sparse.fit(transactions)
    fit_time_sparse = time.perf_counter() - t0

    t0_tr = time.perf_counter()
    X_tr_sparse = pipe_sparse.transform(transactions)
    transform_time_sparse = time.perf_counter() - t0_tr

    rules_sparse = pipe_sparse.named_steps["rules"].rules_
    feature_names_sparse = pipe_sparse.named_steps["rules"].get_feature_names_out()

    results["sparse_mode"] = {
        "fit_time_s": round(fit_time_sparse, 4),
        "transform_time_s": round(transform_time_sparse, 4),
        "n_itemsets": len(pipe_sparse.named_steps["rules"].frequent_itemsets_),
        "n_rules": len(rules_sparse),
        "transform_shape": list(X_tr_sparse.shape),
        "feature_names_count": len(feature_names_sparse),
        "status": pipe_sparse.named_steps["rules"].status_,
        "is_degraded": pipe_sparse.named_steps["rules"].is_degraded_,
    }
    print(f"  Sparse Pipeline: {len(rules_sparse)} rules discovered in {fit_time_sparse:.4f}s, "
          f"transform shape: {X_tr_sparse.shape}")

    # 1.2 Dense Mode Pipeline (evaluated on 2000 transaction subset for bounded memory)
    subset_tx = transactions[:2000]
    t0 = time.perf_counter()
    pipe_dense = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=False)),
        ("rules", AssociationRulesEstimator(
            min_support=0.02,
            metric="confidence",
            min_threshold=0.25,
            algorithm="fpgrowth",
            sparse_output=False,
        )),
    ])
    pipe_dense.fit(subset_tx)
    fit_time_dense = time.perf_counter() - t0

    t0_tr = time.perf_counter()
    X_tr_dense = pipe_dense.transform(subset_tx)
    transform_time_dense = time.perf_counter() - t0_tr

    rules_dense = pipe_dense.named_steps["rules"].rules_
    feature_names_dense = pipe_dense.named_steps["rules"].get_feature_names_out()

    results["dense_mode"] = {
        "fit_time_s": round(fit_time_dense, 4),
        "transform_time_s": round(transform_time_dense, 4),
        "n_itemsets": len(pipe_dense.named_steps["rules"].frequent_itemsets_),
        "n_rules": len(rules_dense),
        "transform_shape": list(X_tr_dense.shape),
        "feature_names_count": len(feature_names_dense),
        "status": pipe_dense.named_steps["rules"].status_,
        "is_degraded": pipe_dense.named_steps["rules"].is_degraded_,
    }
    print(f"  Dense Pipeline (N=2000): {len(rules_dense)} rules discovered in {fit_time_dense:.4f}s, "
          f"transform shape: {X_tr_dense.shape}")

    return results


def benchmark_set_output_pandas(data_dir: str) -> Dict[str, Any]:
    """Verify scikit-learn .set_output(transform='pandas') integration."""
    print("\n--- Part 2: .set_output(transform='pandas') Compatibility ---")

    groceries_path = os.path.join(data_dir, "groceries.csv")
    transactions = load_transactions(groceries_path)[:500]

    # Test on standalone TransactionEncoder
    enc = TransactionEncoder(sparse_output=False)
    enc.set_output(transform="pandas")
    df_encoded = enc.fit_transform(transactions)
    is_enc_df = isinstance(df_encoded, pd.DataFrame)
    enc_feature_names = enc.get_feature_names_out()
    enc_cols_match = list(df_encoded.columns) == list(enc_feature_names)

    # Test on standalone AssociationRulesEstimator
    est = AssociationRulesEstimator(
        min_support=0.03,
        metric="confidence",
        min_threshold=0.20,
        sparse_output=False,
    )
    est.set_output(transform="pandas")
    est.fit(df_encoded)
    df_rules = est.transform(df_encoded)
    is_rules_df = isinstance(df_rules, pd.DataFrame)
    rule_feature_names = est.get_feature_names_out()
    rules_cols_match = list(df_rules.columns) == list(rule_feature_names)

    # Test on full Pipeline
    pipe = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=False)),
        ("rules", AssociationRulesEstimator(min_support=0.03, min_threshold=0.20, sparse_output=False)),
    ])
    pipe.set_output(transform="pandas")
    pipe.fit(transactions)
    df_pipe_out = pipe.transform(transactions)
    is_pipe_df = isinstance(df_pipe_out, pd.DataFrame)

    result = {
        "encoder_set_output": {
            "returns_dataframe": is_enc_df,
            "columns_match_feature_names": enc_cols_match,
            "n_columns": df_encoded.shape[1],
            "sample_columns": list(df_encoded.columns[:5]),
        },
        "estimator_set_output": {
            "returns_dataframe": is_rules_df,
            "columns_match_feature_names": rules_cols_match,
            "n_columns": df_rules.shape[1],
            "sample_columns": list(df_rules.columns[:5]) if not df_rules.empty else [],
        },
        "pipeline_set_output": {
            "returns_dataframe": is_pipe_df,
            "shape": list(df_pipe_out.shape),
            "columns_sample": list(df_pipe_out.columns[:5]) if not df_pipe_out.empty else [],
        },
    }

    print(f"  Encoder .set_output(pandas): DataFrame={is_enc_df}, cols_match={enc_cols_match}")
    print(f"  Estimator .set_output(pandas): DataFrame={is_rules_df}, cols_match={rules_cols_match}")
    print(f"  Pipeline .set_output(pandas): DataFrame={is_pipe_df}, shape={df_pipe_out.shape}")

    return result


def benchmark_gridsearchcv() -> Dict[str, Any]:
    """Benchmark GridSearchCV parameter tuning with unsupervised scoring metrics."""
    print("\n--- Part 3: GridSearchCV Hyperparameter Tuning ---")

    # Generate structured synthetic matrix with planted association patterns
    np.random.seed(42)
    n_samples = 300
    n_items = 20
    # Core random background
    X_mat = np.random.choice([0, 1], size=(n_samples, n_items), p=[0.75, 0.25])
    # Plant strong association rules: item 0 -> item 1, and {item 2, item 3} -> item 4
    for i in range(n_samples):
        if X_mat[i, 0] == 1 and np.random.rand() < 0.85:
            X_mat[i, 1] = 1
        if X_mat[i, 2] == 1 and X_mat[i, 3] == 1 and np.random.rand() < 0.90:
            X_mat[i, 4] = 1

    param_grid = {
        "min_support": [0.08, 0.12, 0.18],
        "min_threshold": [0.30, 0.50, 0.70],
    }

    results: Dict[str, Any] = {
        "param_grid": param_grid,
        "n_samples": n_samples,
        "n_items": n_items,
        "scorers": {},
    }

    scorers = [
        ("mean_confidence", mean_confidence_scorer),
        ("total_support", total_support_scorer),
        ("mean_lift", mean_lift_scorer),
    ]

    for scorer_name, scorer_fn in scorers:
        t0 = time.perf_counter()
        grid = GridSearchCV(
            estimator=AssociationRulesEstimator(
                algorithm="eclat",
                metric="confidence",
                sparse_output=False,
            ),
            param_grid=param_grid,
            cv=3,
            scoring=scorer_fn,
            return_train_score=True,
        )
        # Unsupervised grid search: y can be None or dummy array
        grid.fit(X_mat, np.zeros(n_samples))
        tuning_time = time.perf_counter() - t0

        cv_results = grid.cv_results_
        candidates_evaluated = len(cv_results["params"])

        results["scorers"][scorer_name] = {
            "best_params": grid.best_params_,
            "best_score": round(float(grid.best_score_), 6),
            "tuning_time_s": round(tuning_time, 4),
            "candidates_evaluated": candidates_evaluated,
            "mean_test_scores": [round(float(s), 6) for s in cv_results["mean_test_score"]],
            "params_list": cv_results["params"],
            "best_rules_count": len(grid.best_estimator_.rules_),
            "best_itemsets_count": len(grid.best_estimator_.frequent_itemsets_),
        }
        print(f"  Scorer '{scorer_name}': Best Params={grid.best_params_}, "
              f"Best Score={grid.best_score_:.4f}, Rules={len(grid.best_estimator_.rules_)} in {tuning_time:.3f}s")

    return results


def benchmark_downstream_classification(data_dir: str) -> Dict[str, Any]:
    """Benchmark Pipeline([('encoder', ...), ('rules', ...), ('clf', LogisticRegression())]) on real data."""
    print("\n--- Part 4: Downstream Transactional Classification Benchmark ---")

    benchmarks: Dict[str, Any] = {}

    # 4.1 Tic-Tac-Toe Endgame Classification
    # Task: Predict whether game outcome is positive (win for x) or negative
    ttt_path = os.path.join(data_dir, "tic_tac_toe.dat")
    with open(ttt_path, "r", encoding="utf-8") as f:
        ttt_lines = [line.strip().split() for line in f if line.strip()]

    X_ttt = []
    y_ttt = []
    for tokens in ttt_lines:
        features = [t for t in tokens if not t.startswith("a9=")]
        target = 1 if "a9=positive" in tokens else 0
        X_ttt.append(features)
        y_ttt.append(target)

    y_ttt = np.array(y_ttt)
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_ttt, y_ttt, test_size=0.30, random_state=42, stratify=y_ttt
    )

    # 4.1.1 Rule-Based Pipeline
    t0 = time.perf_counter()
    pipe_ttt = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=False)),
        ("rules", AssociationRulesEstimator(
            min_support=0.08,
            metric="confidence",
            min_threshold=0.50,
            algorithm="apriori",
            sparse_output=False,
        )),
        ("clf", LogisticRegression(solver="liblinear", random_state=42, max_iter=500)),
    ])
    pipe_ttt.fit(X_train_t, y_train_t)
    fit_time_ttt = time.perf_counter() - t0

    y_pred_train_t = pipe_ttt.predict(X_train_t)
    y_pred_test_t = pipe_ttt.predict(X_test_t)
    n_rules_ttt = len(pipe_ttt.named_steps["rules"].rules_)

    # 4.1.2 Raw Item Baseline (Direct One-Hot without Association Rules)
    t0_b = time.perf_counter()
    base_ttt = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=False)),
        ("clf", LogisticRegression(solver="liblinear", random_state=42, max_iter=500)),
    ])
    base_ttt.fit(X_train_t, y_train_t)
    base_pred_test_t = base_ttt.predict(X_test_t)
    base_acc_ttt = accuracy_score(y_test_t, base_pred_test_t)

    benchmarks["tic_tac_toe"] = {
        "n_samples": len(X_ttt),
        "n_train": len(X_train_t),
        "n_test": len(X_test_t),
        "discovered_rules": n_rules_ttt,
        "fit_time_s": round(fit_time_ttt, 4),
        "pipeline_metrics": {
            "train_accuracy": round(float(accuracy_score(y_train_t, y_pred_train_t)), 4),
            "test_accuracy": round(float(accuracy_score(y_test_t, y_pred_test_t)), 4),
            "test_precision": round(float(precision_score(y_test_t, y_pred_test_t, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test_t, y_pred_test_t, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_test_t, y_pred_test_t, zero_division=0)), 4),
        },
        "raw_item_baseline_accuracy": round(float(base_acc_ttt), 4),
        "accuracy_gain": round(float(accuracy_score(y_test_t, y_pred_test_t) - base_acc_ttt), 4),
    }
    print(f"  Tic-Tac-Toe Classification:")
    print(f"    Discovered rules: {n_rules_ttt}")
    print(f"    Pipeline Test Accuracy: {accuracy_score(y_test_t, y_pred_test_t):.4f} "
          f"(vs Raw Baseline: {base_acc_ttt:.4f}, Gain: {accuracy_score(y_test_t, y_pred_test_t) - base_acc_ttt:+.4f})")
    print(f"    F1 Score: {f1_score(y_test_t, y_pred_test_t):.4f}")

    # 4.2 Mushroom Poisonous vs Edible Classification
    # Classic FIMI encoding: '1' = poisonous, '2' = edible
    mush_path = os.path.join(data_dir, "mushroom.dat")
    with open(mush_path, "r", encoding="utf-8") as f:
        mush_lines = [line.strip().split() for line in f if line.strip()]

    X_mush = []
    y_mush = []
    for tokens in mush_lines:
        # Separate class label: '1' is poisonous, '2' is edible
        features = [t for t in tokens if t not in ("1", "2")]
        target = 1 if "1" in tokens else 0
        X_mush.append(features)
        y_mush.append(target)

    y_mush = np.array(y_mush)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_mush, y_mush, test_size=0.30, random_state=42, stratify=y_mush
    )

    t0 = time.perf_counter()
    pipe_mush = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=False)),
        ("rules", AssociationRulesEstimator(
            min_support=0.35,
            metric="confidence",
            min_threshold=0.60,
            algorithm="fpgrowth",
            sparse_output=False,
        )),
        ("clf", LogisticRegression(solver="liblinear", random_state=42, max_iter=500)),
    ])
    pipe_mush.fit(X_train_m, y_train_m)
    fit_time_mush = time.perf_counter() - t0

    y_pred_train_m = pipe_mush.predict(X_train_m)
    y_pred_test_m = pipe_mush.predict(X_test_m)
    n_rules_mush = len(pipe_mush.named_steps["rules"].rules_)

    base_mush = Pipeline([
        ("encoder", TransactionEncoder(sparse_output=False)),
        ("clf", LogisticRegression(solver="liblinear", random_state=42, max_iter=500)),
    ])
    base_mush.fit(X_train_m, y_train_m)
    base_pred_test_m = base_mush.predict(X_test_m)
    base_acc_mush = accuracy_score(y_test_m, base_pred_test_m)

    benchmarks["mushroom"] = {
        "n_samples": len(X_mush),
        "n_train": len(X_train_m),
        "n_test": len(X_test_m),
        "discovered_rules": n_rules_mush,
        "fit_time_s": round(fit_time_mush, 4),
        "pipeline_metrics": {
            "train_accuracy": round(float(accuracy_score(y_train_m, y_pred_train_m)), 4),
            "test_accuracy": round(float(accuracy_score(y_test_m, y_pred_test_m)), 4),
            "test_precision": round(float(precision_score(y_test_m, y_pred_test_m, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_test_m, y_pred_test_m, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_test_m, y_pred_test_m, zero_division=0)), 4),
        },
        "raw_item_baseline_accuracy": round(float(base_acc_mush), 4),
        "accuracy_gain": round(float(accuracy_score(y_test_m, y_pred_test_m) - base_acc_mush), 4),
    }
    print(f"\n  Mushroom Classification:")
    print(f"    Discovered rules: {n_rules_mush}")
    print(f"    Pipeline Test Accuracy: {accuracy_score(y_test_m, y_pred_test_m):.4f} "
          f"(vs Raw Baseline: {base_acc_mush:.4f}, Gain: {accuracy_score(y_test_m, y_pred_test_m) - base_acc_mush:+.4f})")
    print(f"    F1 Score: {f1_score(y_test_m, y_pred_test_m):.4f}")

    return benchmarks


def run_pipeline_benchmarks(data_dir: str = DEFAULT_DATA_DIR, output_path: str = DEFAULT_OUTPUT_PATH) -> Dict[str, Any]:
    """Execute complete Pipeline and Scikit-Learn integration benchmark suite."""
    print("==========================================================================")
    print("  SKARM PIPELINE & SCIKIT-LEARN INTEGRATION BENCHMARK SUITE")
    print("==========================================================================")

    results = {
        "metadata": {
            "title": "SKARM Scikit-Learn Integration & Pipeline Evaluation (RQ4)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": f"{platform.machine()}, {platform.system()} {platform.release()}",
            "python_version": sys.version.split()[0],
        },
        "pipeline_basics": benchmark_pipeline_basics(data_dir),
        "set_output_pandas": benchmark_set_output_pandas(data_dir),
        "gridsearchcv": benchmark_gridsearchcv(),
        "downstream_classification": benchmark_downstream_classification(data_dir),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Pipeline benchmark results saved to: {output_path}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="SKARM Pipeline & Scikit-Learn Integration Benchmark")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for results JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    run_pipeline_benchmarks(data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
