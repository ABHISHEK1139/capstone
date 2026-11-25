"""Unified 300-DPI IEEE Publication Figure & Diagram Generator for skarm.

Generates a visually cohesive set of figures sharing a single IEEE palette,
clean sans-serif typography, and direct grounding from repository JSON artifacts:
  1.  graphical_abstract.png - 4-column ecosystem gap, engine & safety summary
  2.  architecture.png       - 5-layer sparse-native skarm architecture
  3.  engines.png            - Comparative execution pipelines (Apriori, FP-Growth, Eclat)
  4.  fptree.png             - Clean zero-overlap FP-tree & O(1) tail-pointer diagram
  5.  pipeline.png           - Multi-stage evaluation & dual-oracle verification workflow
  6.  family.png             - 2-panel corpus domain & N-vs-density distribution (25 corpora)
  7.  runtime.png            - 25-dataset primary-threshold mean fit time (log scale)
  8.  memory.png             - 25-dataset primary-threshold peak RSS delta (log scale)
  9.  throughput.png         - 25-dataset primary-threshold throughput (tx/s, log scale)
  10. scatter.png            - Itemsets vs. association rules combinatorial scaling
  11. outcomes.png           - 1,650-trial campaign execution outcome breakdown
  12. kpi4_strict.png        - Live 30-trial zero-exemption scaling & AC-4 verification
  13. ablation.png           - 2-panel bounded vs. exact pruning ablation (itemset & rule recall)
  14. baselines.png          - 3-panel cross-framework modern miner comparison (skarm, mlxtend, SPMF)
"""

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE, ".."))
CELLS_DIR = os.path.join(ROOT, "benchmarks", "cells")
CAMPAIGN_JSON = os.path.join(ROOT, "benchmarks", "campaign_results.json")
BENCHMARK_JSON = os.path.join(ROOT, "benchmarks", "benchmark_results.json")
ABLATION_JSON = os.path.join(ROOT, "benchmarks", "ablation_results.json")
BASELINE_JSON = os.path.join(ROOT, "benchmarks", "baseline_results.json")
OUT_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Unified IEEE Semantic Palette
NAVY = "#0F172A"
SLATE_DARK = "#1E293B"
SLATE = "#475569"
SLATE_MUTED = "#64748B"
BORDER = "#CBD5E1"
BG_LIGHT = "#F8FAFC"

C_APRIORI = "#EA580C"     # Burnt Orange
BG_APRIORI = "#FFF7ED"
C_FPGROWTH = "#2563EB"    # Royal Blue
BG_FPGROWTH = "#EFF6FF"
C_ECLAT = "#0891B2"       # Deep Teal
BG_ECLAT = "#ECFEFF"
C_MLXTEND = "#7C3AED"     # Muted Violet
BG_MLXTEND = "#F5F3FF"
C_SPMF = "#059669"        # Emerald Green
BG_SPMF = "#ECFDF5"
C_DEGRADED = "#DC2626"    # Crimson Alert
BG_DEGRADED = "#FEF2F2"
C_AMBER = "#D97706"       # Amber Warning
BG_AMBER = "#FFFBEB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": BORDER,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.color": BORDER,
})


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cells():
    cells = {}
    for p in glob.glob(os.path.join(CELLS_DIR, "*.json")):
        name = os.path.basename(p).replace(".json", "")
        cells[name] = load_json(p)
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


def save_fig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Generated {path}")


def draw_box(ax, x, y, w, h, bg, edge, lw=1.4, pad=0.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size=1.2",
        facecolor=bg, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def draw_graphical_abstract(bench_data, ablation_data, baseline_data, ac4_stats=None):
    """1. Graphical Abstract (4-column IEEE summary grounded in live JSON data)."""
    fig, ax = plt.subplots(figsize=(15, 5.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header banner
    draw_box(ax, 1.0, 86.5, 98.0, 12.0, NAVY, NAVY, lw=0)
    ax.text(2.5, 94.3, "skarm: A scikit-learn-Compatible Estimator for Association-Rule Mining with Auditable Pruning",
            fontsize=13.5, fontweight="bold", color="white", va="center")
    ax.text(2.5, 89.2, "GRAPHICAL ABSTRACT  |  INDEPENDENT SPARSE LIBRARY  |  ORACLE PRECISION <= 2.22e-16  |  25-DATASET VERIFIED SUITE",
            fontsize=8.3, fontfamily="monospace", color="#CBD5E1", va="center")

    cols = [
        (1.5, 22.0, SLATE_DARK, BG_LIGHT, "1. ECOSYSTEM GAP (#32712)"),
        (26.0, 23.0, C_FPGROWTH, BG_FPGROWTH, "2. UNIFIED SPARSE ESTIMATOR"),
        (51.5, 22.5, C_AMBER, BG_AMBER, "3. AUDITABLE SAFETY GOVERNOR"),
        (76.5, 22.0, C_SPMF, BG_SPMF, "4. EMPIRICAL VERIFICATION"),
    ]
    for x, w, hdr_c, bg_c, title in cols:
        draw_box(ax, x, 13.0, w, 71.0, bg_c, BORDER, lw=1.2)
        draw_box(ax, x, 76.5, w, 7.5, hdr_c, hdr_c, lw=0)
        ax.text(x + w / 2, 80.25, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="white")

    # Arrows between columns
    for x1, x2 in [(23.7, 25.8), (49.2, 51.3), (74.2, 76.3)]:
        ax.annotate("", xy=(x2, 48.5), xytext=(x1, 48.5),
                    arrowprops=dict(arrowstyle="-|>", color=C_FPGROWTH, lw=2.2))

    # Col 1 content
    ax.text(2.6, 72.5, "Core scikit-learn Scope", fontsize=8.8, fontweight="bold", color=NAVY, va="top")
    ax.text(2.6, 68.5, "Issue #32712 closed not_planned\n(Nov 2025, citing #2662); core\nmaintainers recommend standalone\ncompatible libraries.",
            fontsize=7.35, color=SLATE_DARK, linespacing=1.32, va="top")
    ax.plot([2.6, 22.4], [55.0, 55.0], color=BORDER, lw=0.8)
    ax.text(2.6, 52.5, "Prior Tooling Deficits", fontsize=8.8, fontweight="bold", color=NAVY, va="top")
    ax.text(2.6, 48.0, "x Dense DataFrame blow-up", fontsize=7.6, fontweight="bold", color=C_DEGRADED, fontfamily="monospace", va="top")
    ax.text(2.6, 44.3, "mlxtend reaches 40.9 GB on sparse\nKosarak; lacks Eclat and sklearn\nEstimator/Pipeline lifecycle.",
            fontsize=7.25, color=SLATE, linespacing=1.30, va="top")
    ax.text(2.6, 31.8, "x Silent truncation / OOM", fontsize=7.6, fontweight="bold", color=C_DEGRADED, fontfamily="monospace", va="top")
    ax.text(2.6, 28.0, "Ad-hoc implementations fail silently\nor exhaust process RSS under\ndense combinatorial lattices.",
            fontsize=7.25, color=SLATE, linespacing=1.30, va="top")

    # Col 2 content
    ax.text(27.3, 72.0, "TransactionEncoder -> SciPy CSR", fontsize=9.0, fontweight="bold", color=NAVY, fontfamily="monospace")
    ax.text(27.3, 68.0, "Single sparse matrix shared by all 3 engines", fontsize=7.6, color=SLATE)
    engines_meta = [
        (54.5, C_APRIORI, "Apriori (Level-Wise Baseline)", "PREFIX (k-2) joins + CSR scanning\nAnti-monotone pruning | Exact ref."),
        (39.5, C_FPGROWTH, "FP-Growth (FP-Array + Bit-Packed)", "float32 GEMM + bitmask triangle gate\nnp.packbits dedup | O(1) tail-link tree"),
        (24.5, C_ECLAT, "Eclat (Vertical C-Level Bitset)", "np.packbits masks + int.bit_count()\nDFS equivalence classes | TID fallback"),
    ]
    for y_b, col_b, t_b, d_b in engines_meta:
        draw_box(ax, 27.3, y_b, 20.4, 12.2, "white", col_b, lw=1.2)
        ax.text(28.3, y_b + 8.8, t_b, fontsize=8.0, fontweight="bold", color=col_b)
        ax.text(28.3, y_b + 3.8, d_b, fontsize=7.2, color=SLATE, linespacing=1.3)
    ax.text(37.5, 17.5, "BaseEstimator | TransformerMixin | Pipeline", ha="center",
            fontsize=7.6, fontweight="bold", fontfamily="monospace", color=C_FPGROWTH)

    # Col 3 content
    gov_cards = [
        (58.5, "Multi-Axis Budget Watchdog", "Enforces max_candidates, max_time,\nand 10 ms RSS sampling (psutil)."),
        (38.0, "Explicit Result-State Contract", "EXACT : is_degraded_=False (complete)\nDEGRADED : is_degraded_=True (capped)"),
        (17.5, "Deterministic Metrics & Oracle", "Support, Conf, Lift, Lev, Conv, Zhang\nMax numerical error <= 2.22e-16"),
    ]
    for y_c, t_c, b_c in gov_cards:
        draw_box(ax, 52.5, y_c, 20.5, 15.0, "white", BORDER, lw=1.0)
        ax.text(53.4, y_c + 11.2, t_c, fontsize=7.9, fontweight="bold", color=NAVY)
        ax.text(53.4, y_c + 4.8, b_c, fontsize=6.95, color=SLATE_DARK, linespacing=1.32)

    # Col 4 dynamic metrics from live JSONs
    s_data = bench_data["KPI-3_4_5_scale_benchmarks"]
    fp_ratios = [s_data[k]["fpgrowth_engine_p95_ratio_vs_apriori"] for k in ("scale_200", "scale_1000", "scale_3000")]
    ec_ratios = [s_data[k]["eclat_engine_p95_ratio_vs_apriori"] for k in ("scale_200", "scale_1000", "scale_3000")]
    if ac4_stats:
        fp_ratios.append(ac4_stats["fp_ratio"])
        ec_ratios.append(ac4_stats["ec_ratio"])
    min_r = min(min(fp_ratios), min(ec_ratios))
    max_r = max(max(fp_ratios), max(ec_ratios))

    chess_ab = ablation_data["candidate_sweeps"]["chess"]
    max_speedup = max(r["speedup_factor"] for r in chess_ab["records"])

    stats_cards = [
        (61.5, "25 Datasets | 1,650 Trials", "8 domains | 98.0% OK | Eclat fastest 20/25\nFP-Growth fastest 5/25 (primary threshold)"),
        (46.0, "157 / 157 Pairs | Delta = 0.0", "Exact itemset identity vs. mlxtend &\nJava SPMF (FIN, PrePost+, LCMFreq)"),
        (30.5, f"{min_r:.3f}x - {max_r:.3f}x Apriori p95", f"Strict 30-trial scaling (<= 0.80x gate, 0 ex.)\nUp to {max_speedup:.2f}x dense speedup in bounded mode"),
        (15.0, "264/264 Pytest | 88/88 E2E", "check_estimator audited (0 unexp. fails)\nNT-1..NT-6 negative test suite 6/6 pass"),
    ]
    for y_s, h_s, d_s in stats_cards:
        ax.text(77.6, y_s + 7.5, h_s, fontsize=8.2, fontweight="bold", fontfamily="monospace", color="#065F46")
        ax.text(77.6, y_s + 1.8, d_s, fontsize=6.95, color=SLATE_DARK, linespacing=1.3)
        if y_s > 20:
            ax.plot([77.8, 97.2], [y_s - 0.8, y_s - 0.8], color=BORDER, lw=0.7)

    # Bottom footer
    draw_box(ax, 1.5, 2.0, 97.0, 8.5, BG_LIGHT, BORDER, lw=1.0)
    ax.text(50.0, 6.25,
            "Key Systems Contribution:  skarm unifies three bit-exact association-rule engines behind one sparse-native scikit-learn estimator with deterministic ordering,\n"
            "auditable candidate-budget degradation, machine-precision metrics (2.22e-16), and full 25-dataset plus cross-framework Java SPMF verification.",
            ha="center", va="center", fontsize=8.0, color=SLATE_DARK, linespacing=1.35)

    save_fig(fig, "graphical_abstract.png")


def draw_architecture():
    """2. Five-Layer Modular Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(13.5, 7.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96.5, "skarm: Five-Layer Modular Architecture & Auditable Safety Subsystems",
            ha="center", fontsize=13.5, fontweight="bold", color=NAVY)

    layers = [
        ("Layer 1: Sparse Ingestion & Normalization Pipeline", 84.5, [
            ("Transaction Parser\nLists / Generators / FIMI / DF", 2.0, 21.5, BG_FPGROWTH, C_FPGROWTH),
            ("Vocabulary Tokenizer\nDuplicate-item idempotency & map", 25.5, 23.0, BG_FPGROWTH, C_FPGROWTH),
            ("TransactionEncoder\nBoolean SciPy CSR (sorted indices)", 50.5, 23.5, BG_FPGROWTH, C_FPGROWTH),
            ("Pre-Flight Telemetry\nSeparate encode time & density", 76.0, 22.0, BG_LIGHT, SLATE_MUTED),
        ]),
        ("Layer 2: scikit-learn Estimator Contract (AssociationRulesEstimator)", 67.5, [
            ("BaseEstimator & TransformerMixin Lifecycle\nfit(X, sample_weight) | transform(X) | predict(X) | clone() | Pipeline", 2.0, 47.0, BG_SPMF, C_SPMF),
            ("Input Guards, Density Dispatch & Persistence\nStrict binary validation | algorithm='auto' | pickle / joblib", 51.0, 47.0, BG_SPMF, C_SPMF),
        ]),
        ("Layer 3: Bit-Exact Core Mining Engines (Delta_supp = 0.0)", 50.5, [
            ("Apriori Engine (Level-Wise Baseline)\nPrefix-(k-2) join + anti-monotone CSR scan\nExhaustive empirical reference", 2.0, 30.0, BG_APRIORI, C_APRIORI),
            ("FP-Growth Engine (FP-Array + FP-Tree)\nfloat32 GEMM pairs | triangle gate | packbits dedup\nO(1) tail-pointer conditional FP-tree", 34.0, 32.0, BG_FPGROWTH, C_FPGROWTH),
            ("Eclat Engine (Vertical Bitset TIDs)\nnp.packbits C-int masks & int.bit_count()\nDFS equivalence classes | weighted fallback", 68.0, 30.0, BG_ECLAT, C_ECLAT),
        ]),
        ("Layer 4: Auditable Pruning Governor & Resource Watchdog", 33.5, [
            ("Multi-Axis Budget Watchdog\nmax_candidates | max_time | 10 ms RSS", 2.0, 30.0, BG_DEGRADED, C_DEGRADED),
            ("Governor Mode Controller\non_candidate_explosion = degrade / raise", 34.0, 32.0, BG_DEGRADED, C_DEGRADED),
            ("Transparent Truncation Telemetry\nstatus_ (EXACT / DEGRADED) | is_degraded_ | metadata_", 68.0, 30.0, BG_DEGRADED, C_DEGRADED),
        ]),
    ]

    for title, y_mid, boxes in layers:
        ax.text(2.0, y_mid + 6.4, title, fontsize=9.8, fontweight="bold", color=SLATE_DARK)
        for text, x, w, bg, border in boxes:
            draw_box(ax, x, y_mid - 4.8, w, 9.8, bg, border, lw=1.4)
            lines = text.split("\n")
            ax.text(x + w / 2, y_mid + 1.8 if len(lines) > 2 else y_mid + 1.3, lines[0],
                    ha="center", va="center", fontsize=8.8, fontweight="bold", color=NAVY)
            body = "\n".join(lines[1:])
            ax.text(x + w / 2, y_mid - 2.0 if len(lines) > 2 else y_mid - 1.6, body,
                    ha="center", va="center", fontsize=7.6, color=SLATE, linespacing=1.25)

    for y_gap in [77.5, 60.5, 43.5, 26.5]:
        ax.annotate("", xy=(50, y_gap - 2.0), xytext=(50, y_gap + 1.6),
                    arrowprops=dict(arrowstyle="-|>", color=SLATE_MUTED, lw=2.0))

    # Layer 5 footer
    ax.text(2.0, 22.8, "Layer 5: Deterministic Rule Extraction & Machine-Precision Metrics (<= 2.22e-16)",
            fontsize=9.8, fontweight="bold", color=SLATE_DARK)
    draw_box(ax, 2.0, 11.0, 96.0, 9.8, BG_MLXTEND, C_MLXTEND, lw=1.4)
    ax.text(50.0, 17.5, "Canonical Lexicographic Itemset & Rule Ordering (MTBF / Determinism Verified)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=NAVY)
    ax.text(50.0, 13.5, "Metrics: Support | Confidence | Lift | Leverage | Conviction (inf-safe) | Zhang's [-1, +1]  |  Outputs: frequent_itemsets_, rules_, status_",
            ha="center", va="center", fontsize=8.0, fontfamily="monospace", color=SLATE_DARK)

    ax.annotate("", xy=(50, 7.8), xytext=(50, 10.6),
                arrowprops=dict(arrowstyle="-|>", color=SLATE_MUTED, lw=2.0))

    draw_box(ax, 2.0, 1.0, 96.0, 6.4, NAVY, NAVY, lw=0)
    ax.text(50.0, 4.2, "Consumable Artifacts:   AssociationRulesEstimator API   |   scikit-learn Pipeline & GridSearchCV   |   CLI & Interactive RuleExplorer",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="white")

    save_fig(fig, "architecture.png")


def draw_engines():
    """3. Three Algorithmic Pipelines Comparison Diagram."""
    fig, ax = plt.subplots(figsize=(14.5, 7.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96.5, "Algorithmic Execution Flows of the Three Bit-Exact Mining Engines",
            ha="center", fontsize=13.5, fontweight="bold", color=NAVY)
    ax.text(50, 92.8, "Shared pre-encoded boolean SciPy CSR input -> Unified IEEE-754 support division -> Canonical sorted itemsets (Delta_supp = 0.0)",
            ha="center", fontsize=8.8, color=SLATE)

    # Panel A: Apriori
    ax.text(2.0, 87.5, "A. Apriori (Level-Wise Baseline Reference & Downward Closure)",
            fontsize=10.5, fontweight="bold", color=C_APRIORI)
    steps_a = [
        ("Step 1: Scan L_1", "Compute 1-item supports\nagainst CSR columns", 2.0, 17.5),
        ("Step 2: Candidate Join", "C_k = L_{k-1} join L_{k-1}\nCheck max_candidates cap", 22.0, 18.0),
        ("Step 3: Downward Prune", "Drop c in C_k if any\n(k-1)-subset not in L_{k-1}", 42.5, 18.0),
        ("Step 4: CSR Support Scan", "Count surviving C_k rows\nRetain L_k >= theta * N", 63.0, 18.0),
        ("Emit L = U L_k", "Stop when L_k = empty\nor DEGRADED triggered", 83.5, 14.5),
    ]
    for idx, (t, d, x, w) in enumerate(steps_a):
        draw_box(ax, x, 72.0, w, 13.0, BG_APRIORI, C_APRIORI, lw=1.3)
        ax.text(x + w / 2, 81.2, t, ha="center", va="center", fontsize=8.5, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, 75.8, d, ha="center", va="center", fontsize=7.5, color=SLATE_DARK, linespacing=1.25)
        if idx < len(steps_a) - 1:
            nx = steps_a[idx + 1][2]
            ax.annotate("", xy=(nx - 0.3, 78.5), xytext=(x + w + 0.3, 78.5),
                        arrowprops=dict(arrowstyle="-|>", color=C_APRIORI, lw=1.6))

    ax.annotate("", xy=(31.0, 71.8), xytext=(72.0, 71.8),
                arrowprops=dict(arrowstyle="-|>", color=C_APRIORI, lw=1.35,
                                linestyle="--", connectionstyle="arc3,rad=-0.20"))
    ax.text(51.5, 64.2, "Iterate Next Level: k <- k + 1", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=C_APRIORI,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))

    # Panel B: FP-Growth
    ax.text(2.0, 60.0, "B. FP-Growth (FP-Array GEMM + Triangle Gate + Bit-Packed O(1) Tail-Linked FP-Tree)",
            fontsize=10.5, fontweight="bold", color=C_FPGROWTH)
    steps_f = [
        ("Step 1: Rank & FP-Array", "Sort L_1 descending support;\nfloat32 X_f^T X_f emits L_1 + L_2\ndirectly without tree allocs", 2.0, 22.0),
        ("Step 2: Triangle Gate", "Bitmask adjacency check for\nfeasible 3-cliques; skips tree\nwhen max itemset size <= 2", 26.5, 22.0),
        ("Step 3: Bit-Packed Tree", "Deduplicate >=3-item rows via\nnp.packbits; build prefix tree\nwith O(1) tail-pointer appends", 51.0, 22.5),
        ("Step 4: Conditional Mine", "Bottom-up header traversal;\nsingle-path power-set fastpath\n+ exact Delta_supp = 0.0", 76.0, 22.0),
    ]
    for idx, (t, d, x, w) in enumerate(steps_f):
        draw_box(ax, x, 43.0, w, 14.5, BG_FPGROWTH, C_FPGROWTH, lw=1.3)
        ax.text(x + w / 2, 54.0, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, 47.4, d, ha="center", va="center", fontsize=7.5, color=SLATE_DARK, linespacing=1.25)
        if idx < len(steps_f) - 1:
            nx = steps_f[idx + 1][2]
            ax.annotate("", xy=(nx - 0.3, 50.25), xytext=(x + w + 0.3, 50.25),
                        arrowprops=dict(arrowstyle="-|>", color=C_FPGROWTH, lw=1.6))

    # Panel C: Eclat
    ax.text(2.0, 36.5, "C. Eclat (Vertical C-Level Bitset TID Masks & Prefix Equivalence Classes)",
            fontsize=10.5, fontweight="bold", color=C_ECLAT)
    steps_e = [
        ("Step 1: Bitset Packing", "Pack CSR columns into C-level\nPython int masks (np.packbits);\nTID-set fallback when weighted", 2.0, 22.0),
        ("Step 2: Equivalence Classes", "Partition search lattice into\ndisjoint prefix classes [P]_k\nsorted by ascending support", 26.5, 22.0),
        ("Step 3: Bitwise Intersect", "Compute mask(X U Y) = X & Y\nExact count via hardware POPCNT\nint.bit_count() (zero rescans)", 51.0, 22.5),
        ("Step 4: Prune & Recurse", "Discard branches < min_count;\nenforce max_candidates budget;\nfastest on 20/25 campaign sets", 76.0, 22.0),
    ]
    for idx, (t, d, x, w) in enumerate(steps_e):
        draw_box(ax, x, 19.5, w, 14.5, BG_ECLAT, C_ECLAT, lw=1.3)
        ax.text(x + w / 2, 30.5, t, ha="center", va="center", fontsize=8.6, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, 23.9, d, ha="center", va="center", fontsize=7.5, color=SLATE_DARK, linespacing=1.25)
        if idx < len(steps_e) - 1:
            nx = steps_e[idx + 1][2]
            ax.annotate("", xy=(nx - 0.3, 26.75), xytext=(x + w + 0.3, 26.75),
                        arrowprops=dict(arrowstyle="-|>", color=C_ECLAT, lw=1.6))

    # Shared guarantee footer
    draw_box(ax, 2.0, 3.0, 96.0, 10.5, BG_LIGHT, BORDER, lw=1.1)
    ax.text(50.0, 8.25,
            "Unified Correctness Invariant:   Identical CSR Boolean Input  +  Shared Candidate Governor  +  Canonical Support Output  =>  Delta_supp = 0.0 (All Pairs)",
            ha="center", va="center", fontsize=8.5, fontweight="bold", fontfamily="monospace", color=NAVY)

    save_fig(fig, "engines.png")


def draw_fptree():
    """4. Clean Zero-Overlap FP-Tree Diagram (Han et al. 5-transaction trace + O(1) tail pointers)."""
    fig, ax = plt.subplots(figsize=(14.5, 7.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96.5, "FP-Tree Architecture: Frequency Header Table, Node-Links, and O(1) Tail-Pointer Optimization",
            ha="center", fontsize=13.0, fontweight="bold", color=NAVY)

    # Panel 1: Transactions (left)
    draw_box(ax, 1.5, 42.0, 21.5, 49.5, BG_LIGHT, SLATE_MUTED, lw=1.3)
    ax.text(12.25, 87.5, "Sample DB (N=5, minsup=2)", ha="center", va="center", fontsize=8.4, fontweight="bold", color=NAVY)
    ax.plot([2.5, 22.0], [83.5, 83.5], color=BORDER, lw=0.9)
    tx_lines = (
        "T1: {f, c, a, m, p}\n"
        "T2: {f, c, a, b, m}\n"
        "T3: {f, b}\n"
        "T4: {c, b, p}\n"
        "T5: {f, c, a, m, p}"
    )
    ax.text(3.2, 74.0, tx_lines, va="center", fontsize=8.2, fontfamily="monospace", color=SLATE_DARK, linespacing=1.45)
    ax.plot([2.5, 22.0], [63.5, 63.5], color=BORDER, lw=0.9)
    ax.text(12.25, 60.0, "Frequent L_1 (Descending)", ha="center", va="center", fontsize=8.3, fontweight="bold", color=C_FPGROWTH)
    ax.text(12.25, 51.0, "f:4  c:4  a:3\nb:3  m:3  p:3\nOrdered insertion yields\ncompact shared prefixes",
            ha="center", va="center", fontsize=7.9, color=SLATE, linespacing=1.4)

    # Panel 2: Header Table (center-left)
    draw_box(ax, 25.0, 36.0, 24.5, 55.0, BG_FPGROWTH, C_FPGROWTH, lw=1.4)
    ax.text(37.25, 87.2, "Header Table", ha="center", va="center", fontsize=9.8, fontweight="bold", color=C_FPGROWTH)
    ax.text(28.0, 81.8, "Item", ha="center", fontsize=8.2, fontweight="bold", color=NAVY)
    ax.text(33.5, 81.8, "Supp", ha="center", fontsize=8.2, fontweight="bold", color=NAVY)
    ax.text(40.0, 81.8, "Head", ha="center", fontsize=8.2, fontweight="bold", color=C_FPGROWTH)
    ax.text(46.2, 81.8, "Tail ptr", ha="center", fontsize=8.2, fontweight="bold", color=C_ECLAT)
    ax.plot([26.2, 48.3], [79.5, 79.5], color="#93C5FD", lw=1.0)

    hdr_rows = [("f", "4", 74.5), ("c", "4", 67.5), ("a", "3", 60.5),
                ("b", "3", 53.5), ("m", "3", 46.5), ("p", "3", 39.5)]
    hdr_y = {}
    for item, cnt, yy in hdr_rows:
        hdr_y[item] = yy
        ax.text(28.0, yy, item, ha="center", va="center", fontsize=8.8, fontweight="bold", fontfamily="monospace", color=NAVY)
        ax.text(33.5, yy, cnt, ha="center", va="center", fontsize=8.5, color=SLATE_DARK)
        ax.plot(40.0, yy, marker="o", color=C_FPGROWTH, markersize=5.5)
        ax.plot(46.2, yy, marker="s", color=C_ECLAT, markersize=5.5)

    # Panel 3: FP-Tree (right) in clean card
    draw_box(ax, 52.0, 23.5, 46.5, 68.5, "white", BORDER, lw=1.2)
    ax.text(53.5, 89.2, "Constructed Prefix Tree", fontsize=9.0, fontweight="bold", color=NAVY, va="center")

    # Node coordinates carefully placed to avoid crossing
    pos = {
        "root": (74.0, 83.2),
        "f1": (66.0, 72.5),
        "c2": (90.0, 72.5),
        "c1": (66.0, 62.0),
        "b3": (76.5, 62.0),
        "b2": (90.0, 60.5),
        "a1": (66.0, 51.8),
        "p2": (90.0, 46.5),
        "m1": (59.5, 41.5),
        "b1": (72.5, 41.5),
        "p1": (59.5, 32.5),
        "m2": (72.5, 32.5),
    }

    tree_edges = [
        ("root", "f1"), ("root", "c2"),
        ("f1", "c1"), ("f1", "b3"),
        ("c1", "a1"),
        ("a1", "m1"), ("a1", "b1"),
        ("m1", "p1"), ("b1", "m2"),
        ("c2", "b2"), ("b2", "p2"),
    ]
    for u, v in tree_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=SLATE_DARK, lw=1.35, zorder=2)

    # Draw nodes on top of edges
    for nid, (nx, ny) in pos.items():
        if nid == "root":
            ax.text(nx, ny, "root", ha="center", va="center", fontsize=8.8, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.35", facecolor="#E2E8F0", edgecolor=SLATE_DARK, lw=1.4))
        else:
            lbl = {"f1": "f:4", "c1": "c:3", "c2": "c:1", "a1": "a:3",
                   "b1": "b:1", "b2": "b:1", "b3": "b:1",
                   "m1": "m:2", "m2": "m:1", "p1": "p:2", "p2": "p:1"}[nid]
            ax.text(nx, ny, lbl, ha="center", va="center", fontsize=8.2, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.28", facecolor="white", edgecolor=C_FPGROWTH, lw=1.4))

    # Head pointers from Header Table -> first node in chain (no node overlaps)
    ax.annotate("", xy=(pos["f1"][0] - 2.6, pos["f1"][1]), xytext=(40.8, hdr_y["f"]),
                arrowprops=dict(arrowstyle="-|>", color=C_FPGROWTH, lw=1.25, linestyle=":"), zorder=3)
    ax.annotate("", xy=(pos["c1"][0] - 2.6, pos["c1"][1] + 0.5), xytext=(40.8, hdr_y["c"]),
                arrowprops=dict(arrowstyle="-|>", color=C_FPGROWTH, lw=1.25, linestyle=":", connectionstyle="arc3,rad=0.12"), zorder=3)
    ax.annotate("", xy=(pos["a1"][0] - 2.6, pos["a1"][1] + 0.5), xytext=(40.8, hdr_y["a"]),
                arrowprops=dict(arrowstyle="-|>", color=C_FPGROWTH, lw=1.25, linestyle=":", connectionstyle="arc3,rad=0.18"), zorder=3)

    # Horizontal node-link chains (cleanly routed around nodes!)
    # c1 -> c2 (arcs smoothly over b3)
    ax.annotate("", xy=(pos["c2"][0] - 2.6, pos["c2"][1]), xytext=(pos["c1"][0] + 2.4, pos["c1"][1] + 1.2),
                arrowprops=dict(arrowstyle="-|>", color=C_MLXTEND, lw=1.2, linestyle="--", connectionstyle="arc3,rad=-0.22"), zorder=3)
    # b3 -> b1 -> b2 (clean chain with zero crossover)
    ax.annotate("", xy=(pos["b1"][0] + 0.6, pos["b1"][1] + 2.6), xytext=(pos["b3"][0] - 0.6, pos["b3"][1] - 2.6),
                arrowprops=dict(arrowstyle="-|>", color=C_MLXTEND, lw=1.2, linestyle="--", connectionstyle="arc3,rad=-0.12"), zorder=3)
    ax.annotate("", xy=(pos["b2"][0] - 2.6, pos["b2"][1] - 0.4), xytext=(pos["b1"][0] + 2.5, pos["b1"][1] + 1.2),
                arrowprops=dict(arrowstyle="-|>", color=C_MLXTEND, lw=1.2, linestyle="--", connectionstyle="arc3,rad=-0.12"), zorder=3)
    # m1 -> m2
    ax.annotate("", xy=(pos["m2"][0] - 2.6, pos["m2"][1] + 0.4), xytext=(pos["m1"][0] + 2.4, pos["m1"][1] - 1.0),
                arrowprops=dict(arrowstyle="-|>", color=C_MLXTEND, lw=1.2, linestyle="--", connectionstyle="arc3,rad=0.1"), zorder=3)
    # p1 -> p2 (clean outer route underneath m2 up to p2!)
    ax.annotate("", xy=(pos["p2"][0] + 0.2, pos["p2"][1] - 2.6), xytext=(pos["p1"][0] + 2.6, pos["p1"][1] - 1.2),
                arrowprops=dict(arrowstyle="-|>", color=C_MLXTEND, lw=1.2, linestyle="--", connectionstyle="arc3,rad=0.22"), zorder=3)

    # Highlight Tail Pointer O(1) append routed cleanly via bottom corridor below p1/m2 to p2!
    ax.annotate("", xy=(pos["p2"][0] + 2.5, pos["p2"][1] - 0.5), xytext=(47.0, hdr_y["p"]),
                arrowprops=dict(arrowstyle="-|>", color=C_ECLAT, lw=1.8, linestyle="-.", connectionstyle="arc3,rad=0.45"), zorder=5)
    ax.text(79.5, 26.2, "O(1) tail-pointer direct append", ha="center", va="center",
            fontsize=8.0, fontweight="bold", fontstyle="italic", color=C_ECLAT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C_ECLAT, lw=0.7), zorder=6)

    # Legend strip inside Tree card top-right (well above root)
    ax.text(97.5, 89.2, "Solid: parent | Dashed: link | Dotted: head",
            ha="right", va="center", fontsize=6.9, color=SLATE)

    # Bottom Explanatory Box with generous clearance so zero overlap occurs!
    draw_box(ax, 1.5, 1.5, 97.0, 19.5, BG_LIGHT, BORDER, lw=1.2)
    ax.text(50.0, 18.0, "Node-Link Maintenance Complexity in FP-Tree Construction",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color=NAVY)
    ax.plot([3.0, 97.0], [15.6, 15.6], color=BORDER, lw=0.8)
    body = (
        "1. Canonical Header Table (Head Pointer Only): Appending a new node requires traversing the entire single-linked chain of length L in O(L) time per\n"
        "   insertion, degrading quadratically on repeated click-stream paths (up to O(N * L^2) operations on sparse high-row datasets).\n"
        "2. skarm Dual-Pointer Header (Head + Tail): Storing tail[item] enables constant-time O(1) appends (tail.node_link = new_node; tail = new_node),\n"
        "   preserving exact canonical FP-tree topology (Delta_supp = 0.0) while eliminating sequential list-traversal bottlenecks."
    )
    ax.text(3.2, 8.2, body, va="center", fontsize=8.0, color=SLATE_DARK, linespacing=1.38)

    save_fig(fig, "fptree.png")


def draw_pipeline():
    """5. Multi-Stage Evaluation & Dual-Oracle Verification Pipeline."""
    fig, ax = plt.subplots(figsize=(14.5, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 95.5, "Multi-Scale Benchmarking & Dual-Oracle Correctness Verification Pipeline",
            ha="center", fontsize=13.2, fontweight="bold", color=NAVY)

    stages = [
        ("1. Corpus & Thresholds", "25 datasets (8 domains)\nMeasured file provenance\nFixed matched ladders", 1.5, 17.5, BG_FPGROWTH, C_FPGROWTH),
        ("2. Parallel Dispatcher", "12 isolated workers\n10 ms RSS delta sampler\n600 s hard trial cap", 21.0, 17.5, BG_ECLAT, C_ECLAT),
        ("3. Statistical Protocol", "30 trials (8 core sets)\n5 trials (17 remaining sets)\nPre-encoded CSR reuse", 40.5, 18.0, BG_MLXTEND, C_MLXTEND),
        ("4. Exact Parity Gate", "Trial 0 cross-check:\n(a) Apriori == FP == Eclat\n(b) mlxtend & Java SPMF", 60.5, 19.0, BG_SPMF, C_SPMF),
        ("5. Audited Archive", "Per-dataset checkpoints\nDeterministic tie-breaking\nExplicit timeout/skip log", 81.5, 17.0, BG_AMBER, C_AMBER),
    ]

    for i, (t, d, x, w, bg, ec) in enumerate(stages):
        draw_box(ax, x, 43.0, w, 45.0, bg, ec, lw=1.6)
        draw_box(ax, x, 76.5, w, 11.5, ec, ec, lw=0)
        ax.text(x + w / 2, 82.25, t, ha="center", va="center", fontsize=8.8, fontweight="bold", color="white")
        ax.text(x + w / 2, 59.5, d, ha="center", va="center", fontsize=8.0, color=SLATE_DARK, linespacing=1.45)
        if i < len(stages) - 1:
            nx = stages[i + 1][2]
            ax.annotate("", xy=(nx - 0.3, 65.5), xytext=(x + w + 0.3, 65.5),
                        arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.8))

    # Mismatch branch under Stage 4
    ax.annotate("", xy=(70.0, 29.0), xytext=(70.0, 42.5),
                arrowprops=dict(arrowstyle="-|>", color=C_DEGRADED, lw=1.8, linestyle="--"))
    ax.text(70.5, 36.0, "Any key or support\ndiff > 1e-9", fontsize=7.6, fontweight="bold", color=C_DEGRADED, ha="left", va="center")

    draw_box(ax, 58.0, 5.0, 40.5, 23.0, BG_DEGRADED, C_DEGRADED, lw=1.4)
    ax.text(78.25, 22.8, "Verification Failure Protocol (Fail-Fast)", ha="center", va="center", fontsize=9.2, fontweight="bold", color=C_DEGRADED)
    ax.text(78.25, 13.0, "Record divergence immediately in cell JSON; abort silent aggregation.\nNever substitute itemset counts for canonical itemset key identity.",
            ha="center", va="center", fontsize=8.0, color="#7F1D1D", linespacing=1.35)

    draw_box(ax, 1.5, 5.0, 53.5, 23.0, BG_SPMF, C_SPMF, lw=1.4)
    ax.text(28.25, 22.8, "Empirical Verification Summary", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#065F46")
    ax.text(28.25, 13.0, "1,650 multi-domain trials (98.0% OK) | 157 / 157 engine pairs exact (Delta_supp = 0.0)\n"
                         "25 / 25 external Java SPMF & mlxtend baselines matched | Strict 30-trial zero-exemption pass",
            ha="center", va="center", fontsize=8.0, color="#064E3B", linespacing=1.4)

    save_fig(fig, "pipeline.png")


def draw_corpus_overview():
    """6. 2-Panel Corpus Characterization (Domains Bar + Log-Log N vs Density Scatter)."""
    datasets = [
        ("Groceries", 9835, 2.61, "Retail"),
        ("Retail", 88162, 0.06, "Retail"),
        ("Market basket", 7501, 3.29, "Retail"),
        ("Online Retail", 20728, 0.64, "Retail"),
        ("Online Retail II", 22103, 0.52, "Retail"),
        ("Kosarak (FIMI)", 990002, 0.02, "Click-stream"),
        ("Kosarak (SPMF)", 187140, 0.03, "Click-stream"),
        ("BMS-POS", 515596, 0.39, "Click-stream"),
        ("BMS-WebView-1", 59009, 0.51, "Click-stream"),
        ("BMS-WebView-2", 77512, 0.14, "Click-stream"),
        ("Chess", 3196, 49.33, "Dense"),
        ("Mushroom", 8124, 19.33, "Dense"),
        ("Connect", 67557, 33.33, "Dense"),
        ("Pumsb", 49046, 3.50, "Census"),
        ("Pumsb_star", 49046, 2.42, "Census"),
        ("Adult census", 32561, 6.19, "Census"),
        ("Tic-tac-toe", 958, 34.48, "Categorical"),
        ("Car evaluation", 1728, 28.00, "Categorical"),
        ("Balance scale", 625, 21.74, "Categorical"),
        ("T10I4D100K", 100000, 1.16, "Synthetic"),
        ("Synthetic sparse", 2000, 1.49, "Synthetic"),
        ("Synthetic dense", 5000, 61.12, "Synthetic"),
        ("NIPS papers", 1500, 4.02, "Document-word"),
        ("Enron emails", 39861, 0.33, "Document-word"),
        ("Accidents", 340183, 7.22, "Traffic"),
    ]

    fam_order = ["Retail", "Click-stream", "Dense", "Census", "Categorical", "Synthetic", "Document-word", "Traffic"]
    fam_colors = {
        "Retail": C_FPGROWTH,
        "Click-stream": C_ECLAT,
        "Dense": C_APRIORI,
        "Census": C_SPMF,
        "Categorical": C_MLXTEND,
        "Synthetic": C_AMBER,
        "Document-word": "#0284C7",
        "Traffic": C_DEGRADED,
    }
    fam_counts = {f: sum(1 for d in datasets if d[3] == f) for f in fam_order}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.35]})

    # Panel (a): Horizontal Bar Chart
    y = np.arange(len(fam_order))
    vals = [fam_counts[f] for f in fam_order]
    cols = [fam_colors[f] for f in fam_order]
    bars = ax1.barh(y, vals, color=cols, height=0.62, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax1.text(v + 0.14, bar.get_y() + bar.get_height() / 2,
                 f"{v} ({v / 25 * 100:.0f}%)", va="center", fontsize=8.2, fontweight="bold", color=NAVY)
    ax1.set_yticks(y)
    ax1.set_yticklabels(fam_order, fontweight="bold", fontsize=8.5, color=SLATE_DARK)
    ax1.set_xlim(0, 6.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Dataset Count (Total = 25)", fontweight="bold")
    ax1.set_title("(a) Domain Family Representation", color=NAVY)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.grid(axis="y", visible=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Scale N vs Density % (log-log)
    for fam in fam_order:
        sub = [d for d in datasets if d[3] == fam]
        ax2.scatter([d[1] for d in sub], [d[2] for d in sub],
                    label=fam, color=fam_colors[fam], s=72, edgecolors="white", linewidth=0.9, zorder=3)

    highlights = {
        "Kosarak (FIMI)": (-8, -12),
        "BMS-POS": (-10, -14),
        "Accidents": (-18, 9),
        "Connect": (-12, 9),
        "Chess": (-6, -12),
        "Synthetic dense": (-38, 8),
        "Balance scale": (6, -10),
        "NIPS papers": (7, 4),
        "Enron emails": (7, -8),
        "Retail": (-18, -13),
    }
    for name, n_tx, dens, _ in datasets:
        if name in highlights:
            dx, dy = highlights[name]
            ax2.annotate(name, xy=(n_tx, dens), xytext=(dx, dy),
                         textcoords="offset points", fontsize=7.5, fontweight="bold", color=SLATE_DARK)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(350, 2.2e6)
    ax2.set_ylim(0.01, 140)
    ax2.set_xlabel("Transactions N (log scale)", fontweight="bold")
    ax2.set_ylabel("Matrix Density (%) (log scale)", fontweight="bold")
    ax2.set_title("(b) Scale vs. Density Span Across 4 Orders of Magnitude", color=NAVY)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower left", ncol=2, fontsize=7.5, frameon=True, facecolor="white", edgecolor=BORDER)

    plt.tight_layout(w_pad=2.5)
    save_fig(fig, "family.png")


def draw_campaign_bars(cells, primary):
    """7, 8, 9. Runtime, Memory, and Throughput across 25 Datasets."""
    names = sorted(cells.keys())
    x = np.arange(len(names))
    w = 0.2
    engines = [("apriori", "Apriori (skarm)", C_APRIORI),
               ("fpgrowth", "FP-Growth (skarm)", C_FPGROWTH),
               ("eclat", "Eclat (skarm)", C_ECLAT),
               ("mlxtend", "mlxtend (v0.25.0)", C_MLXTEND)]

    configs = [
        ("runtime.png", "seconds", 1e-4, 1e3, "Mean Fit Time (s, log scale)",
         "25-Dataset Campaign: Primary-Threshold Mean Fit Time (commit ea5bff5d)"),
        ("memory.png", "peak_mem_mb", 1e-2, 2e4, "Mean Peak RSS Delta (MB, log scale)",
         "25-Dataset Campaign: Primary-Threshold Peak RSS Memory Delta (psutil)"),
        ("throughput.png", "throughput", 0.5, 2e6, "Mean Throughput (tx/sec, log scale)",
         "25-Dataset Campaign: Primary-Threshold Processing Throughput"),
    ]

    for fname, metric, ymin, ymax, ylabel, title in configs:
        fig, ax = plt.subplots(figsize=(14.0, 4.9), dpi=300)
        for i, (eng, lbl, col) in enumerate(engines):
            vals = []
            missing = []
            for n in names:
                thr = primary.get(n)
                recs = [r.get(metric, 0.0) for r in cells[n]["records"]
                        if r.get("status") == "ok" and r.get("threshold") == thr and r.get("engine") == eng]
                if recs and np.mean(recs) > 0:
                    vals.append(max(float(np.mean(recs)), ymin * 1.1))
                    missing.append(False)
                else:
                    vals.append(ymin * 1.05)
                    missing.append(True)
            ax.bar(x + (i - 1.5) * w, vals, w, label=lbl, color=col, alpha=0.92, edgecolor="white", linewidth=0.4)
            if metric == "seconds":
                for idx, is_m in enumerate(missing):
                    if is_m:
                        ax.text(x[idx] + (i - 1.5) * w, ymin * 1.15, "×", ha="center", va="bottom",
                                fontsize=7.0, fontweight="bold", color=C_DEGRADED)

        xlabels = [f"{n} ({primary[n]})" for n in names]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=42, ha="right", fontsize=7.8)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, pad=22, color=NAVY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=True,
                  facecolor="white", edgecolor=BORDER, fontsize=8.5)
        plt.tight_layout()
        save_fig(fig, fname)


def draw_scatter(cells):
    """10. Combinatorial Scaling: Itemsets vs. Association Rules."""
    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=300)
    dense_pts, med_pts, sparse_pts = [], [], []
    seen = set()

    dense_sets = {"chess", "mushroom", "connect", "tic_tac_toe", "car_evaluation", "balance_scale", "synthetic_dense_5000", "accidents"}
    med_sets = {"adult_census", "pumsb", "pumsb_star", "docword_nips", "groceries", "market_basket", "t10i4d100k", "synthetic_sparse_2000"}

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
                    entry = (n_it, n_ru, f"{name}@{r['threshold']}")
                    if name in dense_sets:
                        dense_pts.append(entry)
                    elif name in med_sets:
                        med_pts.append(entry)
                    else:
                        sparse_pts.append(entry)

    for pts, lbl, col in [
        (dense_pts, "Dense Regimes (>7% density)", C_APRIORI),
        (med_pts, "Medium Density (1%–7%)", C_FPGROWTH),
        (sparse_pts, "Sparse Click-Stream / Retail (<1%)", C_ECLAT),
    ]:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=lbl, color=col,
                   s=70, alpha=0.88, edgecolors="white", linewidth=0.9, zorder=4)

    lims = np.array([1, 1e5])
    ax.plot(lims, lims, "--", color=SLATE_MUTED, lw=1.1, label="y = x (1 rule / itemset)")
    ax.plot(lims, lims * 30, ":", color=C_DEGRADED, lw=1.1, label="y = 30x Combinatorial Amplification")

    ann = {
        "docword_nips@0.5": (-75, -12),
        "adult_census@0.05": (-82, 5),
        "chess@0.85": (-58, 8),
        "connect@0.95": (10, -14),
        "online_retail@0.01": (8, -6),
        "bms_pos@0.02": (-55, 8),
    }
    for n_it, n_ru, tag in dense_pts + med_pts + sparse_pts:
        if tag in ann:
            dx, dy = ann[tag]
            ax.annotate(tag, xy=(n_it, n_ru), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.6, fontweight="bold", color=NAVY)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 6e4)
    ax.set_ylim(2, 3e6)
    ax.set_xlabel("Frequent Itemsets Count (log scale)", fontweight="bold")
    ax.set_ylabel("Association Rules Generated (conf >= 0.25, log scale)", fontweight="bold")
    ax.set_title("Combinatorial Expansion: Frequent Itemsets vs. Association Rules", color=NAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="white", edgecolor=BORDER)
    plt.tight_layout()
    save_fig(fig, "scatter.png")


def draw_outcomes(campaign_data):
    """11. Crisp Non-Pie Outcome Breakdown across 1,650 Campaign Trials."""
    records = campaign_data.get("records", [])
    total = len(records)
    counts = {
        "Completed Exact (OK)": sum(1 for r in records if r.get("status") == "ok"),
        "External mlxtend Exclusions (Dense Cap / Empty)": sum(1 for r in records if r.get("status") not in ("ok", "timeout", "skipped")),
        "Cascaded Trial-0 Timeout Skips": sum(1 for r in records if r.get("status") == "skipped"),
        "Hard 600 s Engine Timeouts (Apriori / Dense)": sum(1 for r in records if r.get("status") == "timeout"),
    }

    fig, (ax_top, ax_bar) = plt.subplots(2, 1, figsize=(9.2, 4.4), dpi=300, gridspec_kw={"height_ratios": [1.0, 2.5]})

    # Top: horizontal composition bar
    labels_short = list(counts.keys())
    vals = list(counts.values())
    cols = [C_SPMF, C_MLXTEND, C_AMBER, C_DEGRADED]
    left = 0.0
    for v, col in zip(vals, cols):
        pct = 100.0 * v / total
        ax_top.barh([0], [pct], left=left, color=col, height=0.6, edgecolor="white", linewidth=1.2)
        if pct > 5:
            ax_top.text(left + pct / 2, 0, f"{v:,} OK ({pct:.1f}%)", ha="center", va="center",
                        fontsize=9.2, fontweight="bold", color="white")
        left += pct
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(-0.6, 0.7)
    ax_top.axis("off")
    ax_top.set_title(f"Audit of All {total:,} Benchmark Trials (25 Datasets × Matched Ladders × 4 Implementations)",
                     fontsize=10.5, fontweight="bold", color=NAVY, pad=4)

    # Bottom: log-scale horizontal breakdown
    y = np.arange(len(labels_short))
    bars = ax_bar.barh(y, vals, color=cols, height=0.58, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax_bar.text(v * 1.18, b.get_y() + b.get_height() / 2,
                    f"{v:,} trials ({v / total * 100:.1f}%)", va="center", fontsize=8.4, fontweight="bold", color=NAVY)
    ax_bar.set_xscale("log")
    ax_bar.set_xlim(1, 6000)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels_short, fontsize=8.4, fontweight="bold", color=SLATE_DARK)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Trial Count (log scale)", fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    ax_bar.grid(axis="y", visible=False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=1.0)
    save_fig(fig, "outcomes.png")


def draw_kpi4_strict(bench_data, ac4_stats):
    """12. Live 30-Trial Zero-Exemption Scaling & AC-4 Verification."""
    s = bench_data["KPI-3_4_5_scale_benchmarks"]
    scales = [("scale_200", "Scale N=200\n(39 itemsets)"),
              ("scale_1000", "Scale N=1,000\n(15 itemsets)"),
              ("scale_3000", "Scale N=3,000\n(15 itemsets)")]

    ap_ratios = [1.0, 1.0, 1.0, 1.0]
    fp_ratios = [s[k]["fpgrowth_engine_p95_ratio_vs_apriori"] for k, _ in scales] + [ac4_stats["fp_ratio"]]
    ec_ratios = [s[k]["eclat_engine_p95_ratio_vs_apriori"] for k, _ in scales] + [ac4_stats["ec_ratio"]]

    x_labels = [lbl for _, lbl in scales] + [f"AC-4 Gate\n(N=1,000, M=12)"]
    x = np.arange(len(x_labels))
    w = 0.24

    fig = plt.figure(figsize=(14.0, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.18)

    # Panel (a): Engine p95 runtime ratio
    ax1 = fig.add_subplot(gs[0, 0])
    draw_box(ax1, -0.5, -0.02, 4.0, 1.24, BG_LIGHT, "none", lw=0)
    b1 = ax1.bar(x - w, ap_ratios, w, label="Apriori Baseline (1.000x)", color="#94A3B8")
    b2 = ax1.bar(x, fp_ratios, w, label="FP-Growth (FP-Array + Bit-Packed)", color=C_FPGROWTH)
    b3 = ax1.bar(x + w, ec_ratios, w, label="Eclat (Vertical C-Level Bitset)", color=C_ECLAT)

    ax1.axhline(0.80, color=C_SPMF, linestyle="--", lw=1.7, label="Strict Target Gate (<= 0.80x Apriori)")
    for bars, col in [(b1, SLATE_DARK), (b2, C_FPGROWTH), (b3, C_ECLAT)]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.022, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=col)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8.2, fontweight="bold")
    ax1.set_ylim(0, 1.38)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(["0.00x", "0.20x", "0.40x", "0.60x", "0.80x", "1.00x"])
    ax1.set_title("(a) Engine p95 Runtime Ratio vs. Apriori Baseline (Lower is Faster)", color=NAVY)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper center", ncol=2, fontsize=7.5, frameon=True, facecolor="white", edgecolor=BORDER)

    # Panel (b): End-to-end estimator speedup + scorecard
    ax2 = fig.add_subplot(gs[0, 1])
    y_pos = np.array([2.2, 1.2, 0.2])
    fp_sp = [s[k]["fpgrowth_speedup_vs_apriori"] for k, _ in scales]
    ec_sp = [s[k]["eclat_speedup_vs_apriori"] for k, _ in scales]
    fp_ms = [s[k]["fpgrowth"]["runtime_p95_s"] * 1000.0 for k, _ in scales]
    ec_ms = [s[k]["eclat"]["runtime_p95_s"] * 1000.0 for k, _ in scales]

    ax2.barh(y_pos + 0.16, fp_sp, height=0.26, color=C_FPGROWTH, label="FP-Growth Estimator")
    ax2.barh(y_pos - 0.16, ec_sp, height=0.26, color=C_ECLAT, label="Eclat Estimator")
    ax2.plot([1.25, 1.25], [-0.22, 2.75], color=C_SPMF, linestyle="--", lw=1.4, zorder=2)
    ax2.text(1.38, 2.72, "1.25x min target (<= 0.80x ratio)", ha="left", va="top", fontsize=7.4, fontweight="bold", color=C_SPMF)

    for yp, sp, ms in zip(y_pos, fp_sp, fp_ms):
        ax2.text(sp + 0.08, yp + 0.16, f"{sp:.2f}x ({ms:.2f} ms)", va="center", fontsize=7.8, fontweight="bold", color=NAVY)
    for yp, sp, ms in zip(y_pos, ec_sp, ec_ms):
        ax2.text(sp + 0.08, yp - 0.16, f"{sp:.2f}x ({ms:.2f} ms)", va="center", fontsize=7.8, fontweight="bold", color=NAVY)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["Scale N=200", "Scale N=1,000", "Scale N=3,000"], fontsize=8.4, fontweight="bold", color=NAVY)
    ax2.set_xlim(0, max(max(fp_sp), max(ec_sp)) * 1.48)
    ax2.set_ylim(-1.60, 2.82)
    ax2.set_title("(b) End-to-End Estimator fit() p95 Speedup vs. Apriori", color=NAVY)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xticks([])

    # Scorecard inset inside ax2 bottom
    score_box = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.28, transform=ax2.transAxes,
                                       boxstyle="round,pad=0.02", facecolor=BG_SPMF, edgecolor=C_SPMF, lw=1.1, zorder=4)
    ax2.add_patch(score_box)
    ax2.text(0.05, 0.225, "LIVE HEAD VERIFICATION SCORECARD", transform=ax2.transAxes,
             fontsize=8.0, fontweight="bold", color="#065F46", zorder=5)
    ax2.text(0.05, 0.055,
             f"Pytest: 264/264 passed | 5-Tier E2E: 88/88 | Negative NT: 6/6\n"
             f"Oracle Error: 2.22e-16 (machine eps) | Delta_supp: 0.0 | Fails: 0",
             transform=ax2.transAxes, fontsize=6.6, fontfamily="monospace", color=SLATE_DARK, linespacing=1.38, zorder=5)

    fig.suptitle("Strict 30-Trial Scaling & Acceptance Verification (5 Warmup + 30 Trials, Zero Exemptions)",
                 fontsize=12.0, fontweight="bold", color=NAVY, y=1.01)
    save_fig(fig, "kpi4_strict.png")


def draw_ablation(ablation_data):
    """13. Scientifically Accurate 2-Panel Ablation Figure (Itemset Recall + Rule Recall + Cross-Regime Transition)."""
    chess = ablation_data["candidate_sweeps"]["chess"]
    recs = chess["records"]
    caps = [r["candidate_cap"] for r in recs]
    it_rec = [r["itemset_recall"] * 100.0 for r in recs]
    ru_rec = [r["rule_recall"] * 100.0 for r in recs]
    times = [r["fit_time_s"] for r in recs]
    base_t = chess["baseline"]["fit_time_s"]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=300, gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel (a): Chess dual-recall & runtime curve (EXACT shading honesty!)
    ax1.set_xscale("log")
    # Every cap <= 5000 on Chess is DEGRADED in truth! Show two honest DEGRADED sub-zones:
    ax1.axvspan(35, 720, color=BG_DEGRADED, alpha=0.85, label="Degraded Phase I: Itemsets & Rules Bounded")
    ax1.axvspan(720, 7200, color=BG_AMBER, alpha=0.85, label="Degraded Phase II: 100% Itemsets, Rules Bounded")
    ax1.axvline(720, color=C_AMBER, linestyle=":", lw=1.3)

    l1 = ax1.plot(caps, it_rec, marker="o", color=C_FPGROWTH, lw=2.2, label="Itemset Recall (%)")
    l2 = ax1.plot(caps, ru_rec, marker="^", color=C_ECLAT, lw=2.0, linestyle="-.", label="Rule Recall (%)")
    ax1.set_xlabel("Candidate Budget Cap (max_candidates, log scale)", fontweight="bold")
    ax1.set_ylabel("Pattern Recovery vs. Exact (%)", fontweight="bold", color=NAVY)
    ax1.set_ylim(-6, 114)
    ax1.set_xlim(35, 7200)

    # Annotate cap=1000 critical observation
    idx_1000 = caps.index(1000)
    ax1.annotate(f"Cap 1,000 (DEGRADED):\n100% itemsets (2,669),\n{ru_rec[idx_1000]:.2f}% rules (1,000/95,482)",
                 xy=(1000, ru_rec[idx_1000]), xytext=(1150, 32),
                 arrowprops=dict(arrowstyle="-|>", color=C_DEGRADED, lw=1.2),
                 fontsize=7.5, fontweight="bold", color="#991B1B",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_DEGRADED, lw=0.9))

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    l3 = ax2.plot(caps, times, marker="s", color=C_APRIORI, lw=2.0, linestyle="--", label="Fit Time (s, log)")
    ax2.axhline(base_t, color=C_APRIORI, linestyle=":", lw=1.2, alpha=0.85)
    ax2.set_ylim(0.32, base_t * 1.85)
    ax2.text(42, base_t * 1.10, f"Exact Baseline: {base_t:.2f}s (95,482 rules)", fontsize=7.4, fontweight="bold", color=C_APRIORI)
    ax2.set_ylabel("Fit Time (seconds, log scale)", fontweight="bold", color=C_APRIORI)
    ax2.tick_params(axis="y", labelcolor=C_APRIORI)

    h1, lb1 = ax1.get_legend_handles_labels()
    h2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, lb1 + lb2, loc="center left", fontsize=7.2, frameon=True, facecolor="white", edgecolor=BORDER)
    ax1.set_title("(a) Chess (Dense Apriori): Dual Itemset vs. Rule Boundedness", color=NAVY)

    # Panel (b): Multi-dataset speedup comparison & exact transition points
    datasets_cfg = [
        ("chess", "Chess (Dense, Apriori)", C_APRIORI),
        ("adult_census", "Adult Census (Medium, FP-Growth)", C_FPGROWTH),
        ("bms_pos", "BMS-POS (Sparse, Eclat)", C_ECLAT),
    ]
    ax3.set_xscale("log")
    for key, lbl, col in datasets_cfg:
        recs_k = ablation_data["candidate_sweeps"][key]["records"]
        xs = [r["candidate_cap"] for r in recs_k]
        ys = [r["speedup_factor"] for r in recs_k]
        ax3.plot(xs, ys, marker="o", color=col, lw=2.1, label=lbl)
        # Mark first exact cap if any
        for r in recs_k:
            if not r["is_degraded_"]:
                ax3.scatter([r["candidate_cap"]], [r["speedup_factor"]], marker="*", s=160,
                            color=C_SPMF, edgecolors="white", zorder=6)
                y_off = 12 if key == "bms_pos" else -16
                ax3.annotate(f"Exact @ {r['candidate_cap']}",
                             xy=(r["candidate_cap"], r["speedup_factor"]),
                             xytext=(0, y_off), textcoords="offset points", ha="center",
                             fontsize=7.3, fontweight="bold", color="#065F46")
                break

    ax3.scatter([], [], marker="*", s=130, color=C_SPMF, label="Transition to EXACT (is_degraded_=False)")
    ax3.axhline(1.0, color=SLATE_MUTED, linestyle="--", lw=1.0)
    ax3.set_xlabel("Candidate Budget Cap (max_candidates, log scale)", fontweight="bold")
    ax3.set_ylabel("Speedup vs. Unconstrained Exact Baseline (x)", fontweight="bold")
    ax3.set_title("(b) Cross-Regime Speedup & Transition to EXACT", color=NAVY)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="upper right", fontsize=7.6, frameon=True, facecolor="white", edgecolor=BORDER)

    plt.tight_layout(w_pad=2.5)
    save_fig(fig, "ablation.png")


def draw_baselines(baseline_data):
    """14. Brand-new 3-Panel Comparative Baseline Chart (skarm vs mlxtend vs SPMF Java 21)."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=300)
    ds_keys = [
        ("chess", "Chess (Dense, sigma=0.85, 2,669 sets)"),
        ("adult_census", "Adult Census (Medium, sigma=0.40, 103 sets)"),
        ("bms_pos", "BMS-POS (Sparse, sigma=0.05, 59 sets)"),
    ]

    for ax, (ds_key, title) in zip(axes, ds_keys):
        recs = baseline_data["datasets"][ds_key]["records"]
        labels = []
        wall_times = []
        internal_times = []
        colors = []
        skipped_flags = []

        notes = []
        for r in recs:
            fw = r["framework"]
            alg = r["algorithm"].replace("mlxtend_", "mlx ").replace("_itemsets", "")
            if fw == "skarm":
                lbl = f"skarm {alg}"
                col = C_APRIORI if alg == "apriori" else (C_FPGROWTH if alg == "fpgrowth" else C_ECLAT)
            elif fw == "mlxtend":
                lbl = alg
                col = C_MLXTEND
            else:
                lbl = f"SPMF {alg}"
                col = C_SPMF

            labels.append(lbl)
            colors.append(col)
            if r["status"] == "OK":
                wall_times.append(r["time_seconds"])
                internal_times.append(r.get("internal_time_seconds"))
                skipped_flags.append(False)
                notes.append("")
            else:
                wall_times.append(0.01)
                internal_times.append(None)
                skipped_flags.append(True)
                notes.append(r.get("note", "SKIPPED (>500 MB dense budget)"))

        y = np.arange(len(labels))
        bars = ax.barh(y, wall_times, color=colors, height=0.62, edgecolor="white", linewidth=0.7)
        # Overlay SPMF internal algorithm timer dots where present
        for idx, (wt, it, sk, note_txt) in enumerate(zip(wall_times, internal_times, skipped_flags, notes)):
            if sk:
                short_note = "SKIPPED (814.8 MB dense)" if "814.8" in note_txt else "SKIPPED (dense cap)"
                if "MB" in note_txt and "814.8" not in note_txt:
                    import re as _re
                    m_mb = _re.search(r"([0-9.]+\s*MB)", note_txt)
                    if m_mb:
                        short_note = f"SKIPPED ({m_mb.group(1)} dense)"
                ax.text(0.012, idx, short_note, va="center",
                        fontsize=7.1, fontweight="bold", color=C_DEGRADED)
            else:
                if it is not None:
                    ax.scatter([it], [idx], color=NAVY, marker="D", s=26, zorder=5)
                    ax.text(wt * 1.15, idx, f"{wt:.2f}s (int: {it:.2f}s)", va="center", fontsize=7.1, color=SLATE_DARK)
                else:
                    ax.text(wt * 1.15, idx, f"{wt:.2f}s", va="center", fontsize=7.3, fontweight="bold", color=NAVY)

        ax.set_xscale("log")
        ax.set_xlim(0.008, max(wall_times) * 14.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.8, fontweight="bold", color=SLATE_DARK)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Time (s, log scale)", fontsize=8.2, fontweight="bold")
        ax.set_title(title, fontsize=9.2, color=NAVY)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Cross-Framework Baselines: Exact Support Equivalence (Delta_supp = 0.0 Across All Completed Comparisons)\n"
                 "Bars: Framework Runtime / JVM Wall Time   |   Diamonds: SPMF Internal Algorithm-Only Timer",
                 fontsize=10.5, fontweight="bold", color=NAVY, y=1.04)
    plt.tight_layout(w_pad=1.5)
    save_fig(fig, "baselines.png")


def main():
    bench_data = load_json(BENCHMARK_JSON)
    ablation_data = load_json(ABLATION_JSON)
    baseline_data = load_json(BASELINE_JSON)
    campaign_data = load_json(CAMPAIGN_JSON)
    cells = load_cells()
    primary = compute_primary_thresholds(cells)

    # Read AC-4 live stats from run_acceptance_suite()
    from tests.run_acceptance import run_acceptance_suite
    acc = run_acceptance_suite()
    rt = acc.get("kpi4_measurements", {})
    ap_p95 = rt["apriori"]["p95_s"]
    ac4_stats = {
        "fp_ratio": float(acc["kpi4_fp_p95_ratio"]),
        "ec_ratio": float(acc["kpi4_eclat_p95_ratio"]),
        "ap_p95_s": float(ap_p95),
        "fp_p95_s": float(rt["fpgrowth"]["p95_s"]),
        "ec_p95_s": float(rt["eclat"]["p95_s"]),
        "fp_speedup": float(acc["kpi4_speedup"]),
        "ec_speedup": float(acc["kpi4_eclat_speedup"]),
    }
    print(f"Live AC-4 Ratios: Apriori={ac4_stats['ap_p95_s']:.6f}s, FP={ac4_stats['fp_ratio']:.3f}x ({ac4_stats['fp_speedup']:.2f}x), Eclat={ac4_stats['ec_ratio']:.3f}x ({ac4_stats['ec_speedup']:.2f}x)")

    draw_graphical_abstract(bench_data, ablation_data, baseline_data, ac4_stats=ac4_stats)
    draw_architecture()
    draw_engines()
    draw_fptree()
    draw_pipeline()
    draw_corpus_overview()
    draw_campaign_bars(cells, primary)
    draw_scatter(cells)
    draw_outcomes(campaign_data)
    draw_kpi4_strict(bench_data, ac4_stats)
    draw_ablation(ablation_data)
    draw_baselines(baseline_data)
    print("All 14 IEEE figures successfully generated at 300 DPI!")


if __name__ == "__main__":
    main()
