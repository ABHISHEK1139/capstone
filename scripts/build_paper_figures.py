"""Generate journal-quality publication figures for skarm paper.
Reads all 25 dataset cell files from benchmarks/cells/*.json and
campaign_results.json, producing publication-grade 300 DPI figures
in paper/figures/:
  1. runtime.png (mean runtime per dataset at primary threshold, log scale)
  2. memory.png (peak memory per dataset at primary threshold, log scale)
  3. throughput.png (transactions/sec per dataset at primary threshold)
  4. itemsets.png (frequent itemset count per dataset at primary threshold)
  5. scatter.png (frequent itemsets vs association rules across all cells)
  6. family.png (dataset distribution across application domains)
  7. outcomes.png (trial execution outcomes across all 1,650 records)
"""
import os
import glob
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

BASE = os.path.dirname(__file__)
CELLS_DIR = os.path.join(BASE, "..", "benchmarks", "cells")
RESULTS_JSON = os.path.join(BASE, "..", "benchmarks", "campaign_results.json")
OUT_DIR = os.path.join(BASE, "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

ENGINES = ["apriori", "fpgrowth", "eclat", "mlxtend"]
COLORS = {
    "apriori": "#D95F02",    # vermilion / rust
    "fpgrowth": "#7570B3",   # purple
    "eclat": "#1B9E77",      # teal
    "mlxtend": "#E7298A",    # magenta / rose
}
LABELS = {
    "apriori": "Apriori (skarm)",
    "fpgrowth": "FP-Growth (skarm)",
    "eclat": "Eclat (skarm)",
    "mlxtend": "mlxtend Baseline",
}


def load_cells():
    cells = {}
    for p in glob.glob(os.path.join(CELLS_DIR, "*.json")):
        name = os.path.basename(p).replace(".json", "")
        with open(p, "r", encoding="utf-8") as f:
            cells[name] = json.load(f)
    return cells


def compute_primary_thresholds(cells):
    primary = {}
    for name, c in cells.items():
        by_thr = defaultdict(lambda: {"apriori": [], "fpgrowth": [], "eclat": [], "mlxtend": []})
        for r in c.get("records", []):
            if r.get("status") == "ok":
                by_thr[r["threshold"]][r["engine"]].append(r)
        if not by_thr:
            primary[name] = None
            continue
        def score(thr):
            cnt = sum(len(by_thr[thr][e]) for e in ["apriori", "fpgrowth", "eclat"])
            return (cnt, -thr)
        primary[name] = max(by_thr.keys(), key=score)
    return primary


def plot_runtime(cells, primary):
    names = sorted(cells.keys())
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)

    for i, eng in enumerate(ENGINES):
        vals = []
        for name in names:
            thr = primary.get(name)
            recs = [r["seconds"] for r in cells[name]["records"]
                    if r.get("status") == "ok" and r.get("threshold") == thr and r.get("engine") == eng]
            vals.append(np.mean(recs) if recs else 1e-4)
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=LABELS[eng], color=COLORS[eng], alpha=0.9)
        # Mark missing / 0 bars
        for idx, v in enumerate(vals):
            if v == 1e-4:
                ax.text(x[idx] + (i - 1.5) * width, 1.2e-4, "×", ha="center", va="bottom", fontsize=7, color="#666666")

    labels = [f"{n}\n({primary[n]})" for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e3)
    ax.set_ylabel("Execution Time (seconds, log scale)")
    ax.set_title("Runtime Comparison across 25 Datasets at Primary Threshold (Log Scale)")
    ax.legend(loc="upper left", ncol=4, frameon=True)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "runtime.png")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_memory(cells, primary):
    names = sorted(cells.keys())
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)

    for i, eng in enumerate(ENGINES):
        vals = []
        for name in names:
            thr = primary.get(name)
            recs = [r.get("peak_mem_mb", 0.0) for r in cells[name]["records"]
                    if r.get("status") == "ok" and r.get("threshold") == thr and r.get("engine") == eng]
            m = np.mean(recs) if recs else 0.0
            vals.append(m if m > 0.01 else 0.01)
        ax.bar(x + (i - 1.5) * width, vals, width, label=LABELS[eng], color=COLORS[eng], alpha=0.9)

    labels = [f"{n}\n({primary[n]})" for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e4)
    ax.set_ylabel("Peak Memory Delta (MB, log scale)")
    ax.set_title("Peak Memory Usage across 25 Datasets at Primary Threshold (Log Scale)")
    ax.legend(loc="upper left", ncol=4, frameon=True)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "memory.png")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_throughput(cells, primary):
    names = sorted(cells.keys())
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)

    for i, eng in enumerate(ENGINES):
        vals = []
        for name in names:
            thr = primary.get(name)
            recs = [r.get("throughput", 0.0) for r in cells[name]["records"]
                    if r.get("status") == "ok" and r.get("threshold") == thr and r.get("engine") == eng]
            tp = np.mean(recs) if recs else 0.0
            vals.append(tp if tp > 0.1 else 0.1)
        ax.bar(x + (i - 1.5) * width, vals, width, label=LABELS[eng], color=COLORS[eng], alpha=0.9)

    labels = [f"{n}\n({primary[n]})" for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 2e6)
    ax.set_ylabel("Throughput (transactions / sec, log scale)")
    ax.set_title("Transactional Throughput across 25 Datasets at Primary Threshold")
    ax.legend(loc="upper right", ncol=4, frameon=True)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "throughput.png")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_itemsets(cells, primary):
    names = sorted(cells.keys())
    x = np.arange(len(names))

    itemset_counts = []
    rule_counts = []
    for name in names:
        thr = primary.get(name)
        recs = [r for r in cells[name]["records"]
                if r.get("status") == "ok" and r.get("threshold") == thr and r.get("engine") == "fpgrowth"]
        if not recs:
            recs = [r for r in cells[name]["records"] if r.get("status") == "ok" and r.get("threshold") == thr]
        if recs:
            itemset_counts.append(max(recs[0].get("n_itemsets", 0), 0.5))
            rule_counts.append(max(recs[0].get("n_rules", 0), 0.5))
        else:
            itemset_counts.append(0.5)
            rule_counts.append(0.5)

    fig, ax = plt.subplots(figsize=(14, 5.0), dpi=300)
    w = 0.35
    ax.bar(x - w/2, itemset_counts, w, label="Frequent Itemsets", color="#2B83BA", alpha=0.9)
    ax.bar(x + w/2, rule_counts, w, label="Association Rules (conf >= 0.25)", color="#D7191C", alpha=0.85)

    labels = [f"{n}\n({primary[n]})" for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 1e6)
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Frequent Itemsets vs Discovered Rules at Primary Support Threshold")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "itemsets.png")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_scatter(cells):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    seen = set()
    pts_x, pts_y, tags = [], [], []

    for name, c in cells.items():
        for r in c.get("records", []):
            if r.get("status") == "ok" and r.get("engine") == "fpgrowth":
                key = (name, r["threshold"])
                if key in seen:
                    continue
                seen.add(key)
                n_it = r.get("n_itemsets", 0)
                n_ru = r.get("n_rules", 0)
                if n_it > 0 and n_ru > 0:
                    pts_x.append(n_it)
                    pts_y.append(n_ru)
                    tags.append(name)

    ax.scatter(pts_x, pts_y, color="#2C7BB6", edgecolors="#000000", alpha=0.75, s=65, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequent Itemsets Count (log scale)")
    ax.set_ylabel("Association Rules Count (conf >= 0.25, log scale)")
    ax.set_title("Itemsets vs Rules Combinatorial Scaling across Benchmark Cells")

    # Add identity / reference diagonal
    lims = [1, 5e5]
    ax.plot(lims, lims, "--", color="#999999", alpha=0.6, label="y = x (equal count)")
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "scatter.png")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_family(cells):
    families = {
        "Retail": ["groceries", "retail", "market_basket", "online_retail", "online_retail_ii"],
        "Dense": ["chess", "mushroom", "connect"],
        "Census": ["pumsb", "pumsb_star", "adult_census"],
        "Click-stream": ["kosarak", "kosarak_spmf", "bms_pos", "bms_webview_1", "bms_webview_2"],
        "Doc-Word": ["docword_nips", "docword_enron"],
        "Categorical": ["tic_tac_toe", "car_evaluation", "balance_scale"],
        "Synthetic": ["t10i4d100k", "synthetic_sparse_2000", "synthetic_dense_5000"],
        "Traffic": ["accidents"],
    }
    counts = {fam: len(members) for fam, members in families.items()}

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    fam_names = list(counts.keys())
    fam_vals = [counts[k] for k in fam_names]
    bar_colors = ["#2B83BA", "#54A3B4", "#7EAE9B", "#A8BE81", "#D1CE67", "#E9AF53", "#E37A42", "#D7191C"]

    bars = ax.barh(fam_names, fam_vals, color=bar_colors, edgecolor="#333333", alpha=0.9)
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.1, b.get_y() + b.get_height()/2, f"{int(w)} datasets", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 7)
    ax.set_xlabel("Number of Benchmarked Datasets")
    ax.set_title("Dataset Distribution across Diverse Application Families (25 Datasets Total)")
    ax.invert_yaxis()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "family.png")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def plot_outcomes():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])

    counts = {"OK": 0, "Timeout": 0, "Skipped": 0, "Error": 0}
    for r in records:
        st = r.get("status", "ok")
        if st == "ok":
            counts["OK"] += 1
        elif st == "timeout":
            counts["Timeout"] += 1
        elif st == "skipped":
            counts["Skipped"] += 1
        else:
            counts["Error"] += 1

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=300)
    cat_names = list(counts.keys())
    cat_vals = [counts[k] for k in cat_names]
    cat_colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"]

    bars = ax.bar(cat_names, cat_vals, color=cat_colors, edgecolor="#222222", width=0.55, alpha=0.9)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 15, f"{h} ({h/len(records)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1850)
    ax.set_ylabel("Number of Trials")
    ax.set_title(f"Trial Execution Outcomes across All {len(records):,} Campaign Trials (PC Benchmark)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "outcomes.png")
    fig.savefig(out)
    plt.close(fig)
    print("Saved", out)


def main():
    print("Loading 25 dataset cell files...")
    cells = load_cells()
    primary = compute_primary_thresholds(cells)
    print("Generating 7 publication figures with 25 datasets (1,650 records)...")
    plot_runtime(cells, primary)
    plot_memory(cells, primary)
    plot_throughput(cells, primary)
    plot_itemsets(cells, primary)
    plot_scatter(cells)
    plot_family(cells)
    plot_outcomes()
    print("All figures successfully generated at 300 DPI!")


if __name__ == "__main__":
    main()
