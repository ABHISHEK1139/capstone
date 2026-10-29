"""Command-line interface for skarm association rule mining."""
import argparse
import sys
import os
import json
import pandas as pd
from skarm.estimator import AssociationRulesEstimator
from skarm.explorer import RuleExplorer
from skarm.datasets import load_groceries, load_chess, load_retail, load_market_basket, load_mushroom, load_fimi_dat, dataset_summary


def main():
    parser = argparse.ArgumentParser(
        description="skarm: Scalable Association-Rule Mining Estimator with Apriori Baseline"
    )
    parser.add_argument("--algorithm", choices=["auto", "apriori", "fpgrowth", "eclat"], default="auto")
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--metric", choices=["confidence", "lift", "support", "leverage", "conviction", "zhangs_metric"], default="confidence")
    parser.add_argument("--min-metric", type=float, default=None)
    parser.add_argument("--max-candidates", type=int, default=100000)
    parser.add_argument("--dataset", choices=["groceries", "chess", "retail", "market_basket", "mushroom"], default=None, help="Use standard real-world benchmark dataset")
    parser.add_argument("--input-csv", type=str, help="Path to CSV containing comma-separated transaction items")
    parser.add_argument("--input-dat", type=str, help="Path to .dat file containing space-separated transaction items")
    parser.add_argument("--output-json", type=str, help="Path to save discovered rules in JSON format")
    parser.add_argument("--top-n", type=int, default=10, help="Number of rules to display")
    parser.add_argument("--demo", action="store_true", help="Run demonstrative market basket transaction analysis")

    args = parser.parse_args()

    threshold = args.min_metric if args.min_metric is not None else args.min_confidence

    transactions = None
    dataset_name = "Synthetic Demonstration"

    if args.dataset == "groceries" or (args.input_csv and "groceries" in args.input_csv.lower()):
        dataset_name = "Groceries Market Basket"
        path = args.input_csv if args.input_csv else "data/groceries.csv"
        transactions = load_groceries(path)
    elif args.dataset == "chess" or (args.input_dat and "chess" in args.input_dat.lower()):
        dataset_name = "Chess Benchmark (Dense)"
        path = args.input_dat if args.input_dat else "data/chess.dat"
        transactions = load_chess(path)
    elif args.dataset == "retail" or (args.input_dat and "retail" in args.input_dat.lower()):
        dataset_name = "Retail Benchmark (Sparse 88k)"
        path = args.input_dat if args.input_dat else "data/retail.dat"
        transactions = load_retail(path)
    elif args.dataset == "market_basket" or (args.input_csv and "market" in args.input_csv.lower()):
        dataset_name = "Market Basket Optimisation (7,501 txns)"
        path = args.input_csv if args.input_csv else "data/market_basket.csv"
        transactions = load_market_basket(path)
    elif args.dataset == "mushroom" or (args.input_dat and "mushroom" in args.input_dat.lower()):
        dataset_name = "Mushroom Benchmark (Dense 8,124 txns)"
        path = args.input_dat if args.input_dat else "data/mushroom.dat"
        transactions = load_mushroom(path)
    elif args.input_csv:
        dataset_name = os.path.basename(args.input_csv)
        transactions = load_groceries(args.input_csv)
    elif args.input_dat:
        dataset_name = os.path.basename(args.input_dat)
        transactions = load_fimi_dat(args.input_dat)
    elif args.demo or not (args.dataset or args.input_csv or args.input_dat):
        dataset_name = "Synthetic Demonstration"
        transactions = [
            ["milk", "bread", "butter"],
            ["bread", "butter"],
            ["milk", "bread"],
            ["beer", "diapers"],
            ["milk", "beer", "diapers", "bread"],
            ["bread", "butter", "milk"],
            ["diapers", "beer"],
            ["milk", "bread", "butter", "eggs"],
        ]

    print(f"\n=======================================================")
    print(f"skarm CLI: Association Rule Mining on {dataset_name}")
    print(f"=======================================================")

    summary = dataset_summary(transactions)
    print(f"Transactions: {summary['n_transactions']:,} | Items: {summary['n_unique_items']:,} | Avg Tx Length: {summary['avg_transaction_length']}")
    print(f"Parameters: Algorithm={args.algorithm.upper()}, min_support={args.min_support}, metric={args.metric}, threshold={threshold}")

    estimator = AssociationRulesEstimator(
        min_support=args.min_support,
        metric=args.metric,
        min_threshold=threshold,
        algorithm=args.algorithm,
        max_candidates=args.max_candidates,
        sparse_output=True,
    )

    estimator.fit(transactions)
    rules = estimator.get_rules()
    explorer = estimator.explore()

    print(f"\nMined {len(estimator.frequent_itemsets_)} frequent itemsets and {len(rules)} association rules:")
    print(explorer.to_markdown(top_n=args.top_n))

    print("\nSummary Statistics:")
    print(json.dumps(explorer.summary_statistics(), indent=2))

    if args.output_json:
        # Export JSON cleanly
        rules_dict = rules.copy()
        if "antecedent_names" in rules_dict.columns:
            rules_dict["antecedent_names"] = rules_dict["antecedent_names"].apply(lambda s: sorted(list(s)))
            rules_dict["consequent_names"] = rules_dict["consequent_names"].apply(lambda s: sorted(list(s)))
        rules_dict["antecedent"] = rules_dict["antecedent"].apply(lambda s: sorted(list(s)))
        rules_dict["consequent"] = rules_dict["consequent"].apply(lambda s: sorted(list(s)))
        rules_dict.to_json(args.output_json, orient="records", indent=2)
        print(f"\nDiscovered rules exported to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
