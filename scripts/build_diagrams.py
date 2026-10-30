"""build_diagrams.py - Generate publication-quality 300 DPI architectural, algorithmic,
and structural diagrams for the skarm research paper.

Produces:
  1. paper/figures/engines.png      - Comparative mechanics of Apriori, FP-Growth, and Eclat
  2. paper/figures/architecture.png - Multi-layer system architecture of the skarm estimator
  3. paper/figures/pipeline.png     - 25-dataset benchmarking, multi-worker scheduling & verification pipeline
  4. paper/figures/fptree.png       - FP-tree with frequency header table & O(1) tail-pointer optimization
  5. paper/figures/ablation.png     - Pruning policy ablation: candidate cap vs recall and fit time
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_engines():
    """Figure 1: Comparative Algorithmic Mechanics."""
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_apriori = "#FFF2E6"
    b_apriori = "#D95F02"
    c_fpgrowth = "#F0EEF8"
    b_fpgrowth = "#7570B3"
    c_eclat = "#E6F5F0"
    b_eclat = "#1B9E77"
    text_dark = "#1A1A1A"
    gray_line = "#4A5568"

    # Panel A: Apriori
    ax.text(2, 93, "A. Apriori (Level-Wise Candidate Generation & Downward Closure)",
            fontsize=12, fontweight="bold", color=b_apriori)

    steps_a = [
        ("Step 1: Scan & Count\nFrequent 1-Itemsets", "Find $L_1 = \{i \\mid \\sigma(i) \\geq \\theta N\}$\nDatabase scan 1", 5, 75, 18, 14),
        ("Step 2: Candidate Join\n$C_k = L_{k-1} \\bowtie L_{k-1}$", "Self-join itemsets sharing\nprefix of length $k-2$", 27, 75, 18, 14),
        ("Step 3: Downward Pruning\n(Anti-Monotone)", "Discard $c \\in C_k$ if any\nsub-itemset $s \\subset c \\notin L_{k-1}$", 49, 75, 18, 14),
        ("Step 4: Matrix Scan\n& Support Counting", "Scan transactions to count\noccurrences of $c \\in C_k$", 71, 75, 18, 14),
        ("Emit Patterns\n$L = \\bigcup L_k$", "Iterate until\n$C_{k+1} = \\emptyset$", 91, 75, 8, 14),
    ]

    for title, desc, x, y, w, h in steps_a:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                     facecolor=c_apriori, edgecolor=b_apriori, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_dark)
        ax.text(x + w/2, y + h*0.25, desc, ha="center", va="center", fontsize=8, color="#4A5568")

    for i in range(len(steps_a) - 1):
        x1 = steps_a[i][2] + steps_a[i][4]
        y1 = steps_a[i][3] + steps_a[i][5]/2
        x2 = steps_a[i+1][2]
        y2 = steps_a[i+1][3] + steps_a[i+1][5]/2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=b_apriori, lw=1.5))

    # Loop back for Apriori
    ax.annotate("Next Level $k \\leftarrow k+1$", xy=(36, 75), xytext=(80, 69),
                arrowprops=dict(arrowstyle="-|>", color=b_apriori, lw=1.2,
                                connectionstyle="arc3,rad=0.25", linestyle="--"),
                fontsize=8.5, color=b_apriori, fontweight="bold", ha="center")

    # Panel B: FP-Growth
    ax.text(2, 60, "B. FP-Growth (Frequent Pattern Tree & Recursive Conditional Mining)",
            fontsize=12, fontweight="bold", color=b_fpgrowth)

    steps_f = [
        ("Step 1: Frequency Ordering", "Scan database; compute item supports;\nsort $L_1$ by descending frequency", 5, 42, 20, 14),
        ("Step 2: 2-Pass FP-Tree Build", "Pass 2: Insert transactions into tree;\nmaintain $O(1)$ tail-linked headers", 29, 42, 22, 14),
        ("Step 3: Conditional Pattern Bases", "For each item in header table (bottom-up):\ntraverse prefix paths via node-links", 55, 42, 22, 14),
        ("Step 4: Recursive Mining", "Construct conditional FP-trees;\nemit itemsets with suffix concatenation", 81, 42, 18, 14),
    ]

    for title, desc, x, y, w, h in steps_f:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                     facecolor=c_fpgrowth, edgecolor=b_fpgrowth, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_dark)
        ax.text(x + w/2, y + h*0.25, desc, ha="center", va="center", fontsize=8, color="#4A5568")

    for i in range(len(steps_f) - 1):
        x1 = steps_f[i][2] + steps_f[i][4]
        y1 = steps_f[i][3] + steps_f[i][5]/2
        x2 = steps_f[i+1][2]
        y2 = steps_f[i+1][3] + steps_f[i+1][5]/2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=b_fpgrowth, lw=1.5))

    # Panel C: Eclat
    ax.text(2, 27, "C. Eclat (Vertical Transaction-ID Layout & Equivalence Class Decomposition)",
            fontsize=12, fontweight="bold", color=b_eclat)

    steps_e = [
        ("Step 1: Vertical Layout", "Transform transactions to inverted list:\neach item mapped to tidset $\\mathcal{T}(i)$", 5, 9, 20, 14),
        ("Step 2: Equivalence Classes", "Partition search space into prefix equivalence\nclasses $[P]_k = \\{X \\mid \\text{prefix}(X)=P\\}$", 29, 9, 22, 14),
        ("Step 3: Depth-First Intersections", "Compute $\\mathcal{T}(X \\cup Y) = \\mathcal{T}(X) \\cap \\mathcal{T}(Y)$;\nfast bitset/sorted array intersection", 55, 9, 22, 14),
        ("Step 4: Support Verification", "Prune if $|\\mathcal{T}| < \\theta N$;\nrecurse on valid sub-classes $[P']$", 81, 9, 18, 14),
    ]

    for title, desc, x, y, w, h in steps_e:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                     facecolor=c_eclat, edgecolor=b_eclat, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_dark)
        ax.text(x + w/2, y + h*0.25, desc, ha="center", va="center", fontsize=8, color="#4A5568")

    for i in range(len(steps_e) - 1):
        x1 = steps_e[i][2] + steps_e[i][4]
        y1 = steps_e[i][3] + steps_e[i][5]/2
        x2 = steps_e[i+1][2]
        y2 = steps_e[i+1][3] + steps_e[i+1][5]/2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=b_eclat, lw=1.5))

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "engines.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {p}")


def draw_architecture():
    """Figure 2: Multi-Layer System Architecture."""
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header title
    ax.text(50, 96, "skarm: Modular Architecture & Component Subsystems",
            ha="center", fontsize=14, fontweight="bold", color="#1E293B")

    # Layers
    layers = [
        ("Layer 1: Input & Ingestion Pipeline", 85, [
            ("Transaction Parser\n(List / Generator / DF)", 3, 20, "#EFF6FF", "#3B82F6"),
            ("Vocabulary Tokenizer\n& Item-to-Index Map", 26, 21, "#EFF6FF", "#3B82F6"),
            ("TransactionEncoder\nSparse CSR / CSC Matrix", 50, 23, "#EFF6FF", "#3B82F6"),
            ("Telemetry Pre-Flight\n(Row/Col Memory Footprint)", 76, 21, "#F8FAFC", "#64748B"),
        ]),
        ("Layer 2: Scikit-learn Estimator Abstraction (AssociationRulesEstimator)", 67, [
            ("scikit-learn API Contract\nfit(X, y=None) | clone() | get_params()", 3, 44, "#F0FDF4", "#16A34A"),
            ("Parameter Validation &\nConfiguration Encapsulation", 50, 47, "#F0FDF4", "#16A34A"),
        ]),
        ("Layer 3: Core Algorithmic Mining Engines", 49, [
            ("Apriori Engine\nLevel-wise CSR scan", 3, 29, "#FFF7ED", "#D95F02"),
            ("FP-Growth Engine\nO(1) Tail-Pointer Header Table", 35, 31, "#F5F3FF", "#7570B3"),
            ("Eclat Engine\nVertical Tidset Intersections", 69, 28, "#ECFDF5", "#1B9E77"),
        ]),
        ("Layer 4: Pruning Governor & Safety System", 31, [
            ("Candidate Budget Watchdog\n(max_candidates cap)", 3, 29, "#FEF2F2", "#DC2626"),
            ("Governor Mode Handler\nExact vs Safe-Degrade", 35, 31, "#FEF2F2", "#DC2626"),
            ("Degraded Mode Signaling\nis_degraded_ & Truncation Telemetry", 69, 28, "#FEF2F2", "#DC2626"),
        ]),
        ("Layer 5: Association Rule & Metric Generation", 13, [
            ("Rule Antecedent / Consequent\nPower-set Decomposition", 3, 31, "#FDF4FF", "#A855F7"),
            ("Numerical Metric Evaluator\nSupport, Confidence, Lift, Leverage", 37, 31, "#FDF4FF", "#A855F7"),
            ("Edge-Case Safeguards\nConviction (inf) | Zhang ([-1,+1])", 71, 26, "#FDF4FF", "#A855F7"),
        ]),
    ]

    for title, y_top, boxes in layers:
        ax.text(3, y_top + 6.5, title, fontsize=10.5, fontweight="bold", color="#334155")
        for b_title, x, w, bg, border in boxes:
            rect = patches.FancyBboxPatch((x, y_top - 5), w, 10, boxstyle="round,pad=0.4",
                                         facecolor=bg, edgecolor=border, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + w/2, y_top, b_title, ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if "API" in b_title or "Engine" in b_title else "normal", color="#1E293B")

    # Connectors between layers
    for y in [79, 61, 43, 25]:
        ax.annotate("", xy=(50, y - 2), xytext=(50, y + 2),
                    arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=2))

    # Output banner
    rect_out = patches.FancyBboxPatch((3, 0), 94, 6, boxstyle="round,pad=0.3",
                                     facecolor="#1E293B", edgecolor="#0F172A", linewidth=1.5)
    ax.add_patch(rect_out)
    ax.text(50, 3, "Deterministic Outputs: frequent_itemsets_ (Lex-Sorted) | rules_ (Audited Metrics) | Telemetry (RSS, Candidates)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#FFFFFF")

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "architecture.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {p}")


def draw_pipeline():
    """Figure 3: Comprehensive 25-Dataset Evaluation & Verification Pipeline."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 95, "Comprehensive 25-Dataset Benchmarking & Verification Pipeline",
            ha="center", fontsize=13, fontweight="bold", color="#0F172A")

    stages = [
        ("1. Benchmark Suite\n& Ladder Config",
         "25 Datasets (8 Domains)\nVerified sizes & densities\nFixed matched thresholds",
         2, 45, 16, 40, "#EFF6FF", "#2563EB"),

        ("2. Multiprocessing\nScheduler",
         "12 Parallel Workers\nIsolated OS processes\npsutil 10ms RSS sampling\n600s hard async timeout",
         21, 45, 17, 40, "#F5F3FF", "#7C3AED"),

        ("3. Statistical\nTrial Dispatcher",
         "8 KPI Datasets: 30 trials\n17 Datasets: 5 trials\nmlxtend baseline: 5 trials\nPre-encoded matrix reuse",
         41, 45, 17, 40, "#FDF2F8", "#DB2777"),

        ("4. Dual-Oracle\nVerification Gate",
         "Trial 0 Strict Audit:\n(a) Skarm cross-engine match\n    Apriori == FP-Growth == Eclat\n(b) mlxtend oracle agreement\n    Max support delta = 0.0",
         61, 45, 19, 40, "#ECFDF5", "#059669"),

        ("5. Checkpoints &\nSynthesis",
         "benchmarks/cells/*.json\nResume-safe persistence\ncampaign_results.json\nPublication-grade figures\n& verified LaTeX tables",
         83, 45, 15, 40, "#FFFBEB", "#D97706"),
    ]

    for title, desc, x, y, w, h, bg, border in stages:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                     facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 6, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#0F172A")
        ax.plot([x + 1, x + w - 1], [y + h - 12, y + h - 12], color=border, lw=1)
        ax.text(x + w/2, y + (h - 12)/2, desc, ha="center", va="center", fontsize=8.2, color="#334155", linespacing=1.3)

    # Connecting arrows
    for i in range(len(stages) - 1):
        x1 = stages[i][2] + stages[i][4]
        y1 = stages[i][3] + stages[i][5]/2
        x2 = stages[i+1][2]
        y2 = stages[i+1][3] + stages[i+1][5]/2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2))

    # Error handling / mismatch branch below Gate
    rect_mismatch = patches.FancyBboxPatch((58, 12), 25, 20, boxstyle="round,pad=0.5",
                                          facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.5)
    ax.add_patch(rect_mismatch)
    ax.text(70.5, 27, "Correctness Gate Violation", ha="center", fontsize=9.5, fontweight="bold", color="#991B1B")
    ax.text(70.5, 19, "Record mismatch in cell report;\nfail immediately; never smooth\nor hide output discrepancy",
            ha="center", fontsize=8, color="#7F1D1D")

    # Mismatch arrow
    ax.annotate("Mismatch Detected", xy=(70.5, 33), xytext=(70.5, 45),
                arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=1.5, linestyle="--"),
                fontsize=8, color="#DC2626", fontweight="bold", ha="center")

    # Results verified banner
    rect_pass = patches.FancyBboxPatch((2, 12), 52, 20, boxstyle="round,pad=0.5",
                                      facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.5)
    ax.add_patch(rect_pass)
    ax.text(28, 27, "Empirical Validation Certificate (PC Campaign)", ha="center", fontsize=9.5, fontweight="bold", color="#166534")
    ax.text(28, 19, "1,650 trial records across 25 datasets (100% complete)\n157 compared engine pairs with 0 mismatches\nEclat fastest on 20/25; FP-Growth on 5/25; Apriori on 0",
            ha="center", fontsize=8, color="#14532D")

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "pipeline.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {p}")


def draw_fptree():
    """Figure 4: Complete FP-Tree with Header Table, Node-Links, and O(1) Tail Optimization."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "FP-Tree Structure: Frequency Header Table, Node-Links & O(1) Tail-Pointer Optimization",
            ha="center", fontsize=13, fontweight="bold", color="#0F172A")

    # Panel 1: Transaction Database Box
    rect_db = patches.FancyBboxPatch((2, 54), 22, 38, boxstyle="round,pad=0.5",
                                    facecolor="#F8FAFC", edgecolor="#64748B", linewidth=1.5)
    ax.add_patch(rect_db)
    ax.text(13, 88, "Sample Transactions\n(minsup = 2)", ha="center", fontsize=9.5, fontweight="bold", color="#1E293B")
    tx_text = (
        "T1: {f, c, a, m, p}\n"
        "T2: {f, c, a, b, m}\n"
        "T3: {f, b}\n"
        "T4: {c, b, p}\n"
        "T5: {f, c, a, m, p}\n"
        "------------------\n"
        "L1 Frequency Order:\n"
        "f:4, c:4, a:3, b:3, m:3, p:3"
    )
    ax.text(13, 68, tx_text, ha="center", fontsize=8.5, color="#334155", fontfamily="monospace")

    # Panel 2: Frequency Header Table
    rect_hdr = patches.FancyBboxPatch((28, 48), 24, 44, boxstyle="round,pad=0.5",
                                     facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.5)
    ax.add_patch(rect_hdr)
    ax.text(40, 88, "Header Table (skarm)", ha="center", fontsize=10, fontweight="bold", color="#1D4ED8")

    # Table columns header
    ax.text(31, 82, "Item", fontsize=8.5, fontweight="bold", color="#1E293B")
    ax.text(37, 82, "Count", fontsize=8.5, fontweight="bold", color="#1E293B")
    ax.text(44, 82, "Head", fontsize=8.5, fontweight="bold", color="#1E293B")
    ax.text(49, 82, "Tail [NEW]", fontsize=8.5, fontweight="bold", color="#DC2626")
    ax.plot([29, 51], [80, 80], color="#93C5FD", lw=1)

    items_hdr = [
        ("f", "4", 76, 76),
        ("c", "4", 71, 71),
        ("a", "3", 66, 66),
        ("b", "3", 61, 61),
        ("m", "3", 56, 56),
        ("p", "3", 51, 51),
    ]

    for item, cnt, y, _ in items_hdr:
        ax.text(31.5, y, item, fontsize=8.5, fontfamily="monospace", color="#1E293B")
        ax.text(38.5, y, cnt, fontsize=8.5, color="#1E293B")
        # Head pointer dot
        ax.plot(45.5, y + 1, marker="o", color="#2563EB", markersize=5)
        # Tail pointer dot
        ax.plot(50.5, y + 1, marker="s", color="#DC2626", markersize=5)

    # Panel 3: FP-Tree Nodes
    # Root
    ax.text(75, 90, "root", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.4", facecolor="#E2E8F0", edgecolor="#475569", lw=1.5))

    # Tree nodes: (label, x, y, id)
    tree_nodes = [
        # Branch 1 (left)
        ("f:4", 67, 78, "f1"),
        ("c:3", 67, 66, "c1"),
        ("a:3", 67, 54, "a1"),
        ("m:2", 63, 42, "m1"),
        ("p:2", 63, 30, "p1"),
        ("b:1", 71, 42, "b1"),
        ("m:1", 71, 30, "m2"),
        # Branch 2 (right of root)
        ("c:1", 85, 78, "c2"),
        ("b:1", 85, 66, "b2"),
        ("p:1", 85, 54, "p2"),
        # Branch 3 (from f1)
        ("b:1", 77, 66, "b3"),
    ]

    node_pos = {}
    for lbl, x, y, nid in tree_nodes:
        node_pos[nid] = (x, y)
        ax.text(x, y, lbl, ha="center", va="center", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="#FFFFFF", edgecolor="#2563EB", lw=1.2))

    # Tree edges (solid black)
    edges = [
        ((75, 87), node_pos["f1"]),
        ((75, 87), node_pos["c2"]),
        (node_pos["f1"], node_pos["c1"]),
        (node_pos["f1"], node_pos["b3"]),
        (node_pos["c1"], node_pos["a1"]),
        (node_pos["a1"], node_pos["m1"]),
        (node_pos["a1"], node_pos["b1"]),
        (node_pos["m1"], node_pos["p1"]),
        (node_pos["b1"], node_pos["m2"]),
        (node_pos["c2"], node_pos["b2"]),
        (node_pos["b2"], node_pos["p2"]),
    ]

    for (x1, y1), (x2, y2) in edges:
        ax.annotate("", xy=(x2, y2 + 2.5), xytext=(x1, y1 - 2.5),
                    arrowprops=dict(arrowstyle="-", color="#334155", lw=1.2))

    # Horizontal node-links (dashed colored arrows from header table and between nodes)
    # Header head pointers:
    ax.annotate("", xy=(node_pos["f1"][0] - 3, node_pos["f1"][1]), xytext=(46.5, 77),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.2, linestyle=":"))
    ax.annotate("", xy=(node_pos["c1"][0] - 3, node_pos["c1"][1]), xytext=(46.5, 72),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.2, linestyle=":"))
    ax.annotate("", xy=(node_pos["a1"][0] - 3, node_pos["a1"][1]), xytext=(46.5, 67),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.2, linestyle=":"))

    # Cross-branch node-links (between nodes)
    # c1 -> c2
    ax.annotate("", xy=(node_pos["c2"][0] - 3, node_pos["c2"][1]), xytext=(node_pos["c1"][0] + 3, node_pos["c1"][1]),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, linestyle="--", connectionstyle="arc3,rad=-0.2"))
    # b3 -> b1 -> b2
    ax.annotate("", xy=(node_pos["b1"][0] + 3, node_pos["b1"][1]), xytext=(node_pos["b3"][0] - 1, node_pos["b3"][1] - 2),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, linestyle="--", connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(node_pos["b2"][0] - 3, node_pos["b2"][1]), xytext=(node_pos["b3"][0] + 3, node_pos["b3"][1]),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, linestyle="--", connectionstyle="arc3,rad=-0.1"))
    # m1 -> m2
    ax.annotate("", xy=(node_pos["m2"][0] - 3, node_pos["m2"][1]), xytext=(node_pos["m1"][0] + 3, node_pos["m1"][1] - 1),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, linestyle="--"))
    # p1 -> p2
    ax.annotate("", xy=(node_pos["p2"][0] - 3, node_pos["p2"][1]), xytext=(node_pos["p1"][0] + 3, node_pos["p1"][1] + 1),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2, linestyle="--", connectionstyle="arc3,rad=-0.3"))

    # Tail pointer direct connection highlight (The Key Engineering Innovation!)
    ax.annotate("", xy=(node_pos["p2"][0] + 3, node_pos["p2"][1]), xytext=(51.5, 52),
                arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=1.5, linestyle="-.", connectionstyle="arc3,rad=-0.4"))

    # Callout Banner for O(1) Tail Pointer
    rect_callout = patches.FancyBboxPatch((2, 2), 96, 20, boxstyle="round,pad=0.5",
                                         facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.5)
    ax.add_patch(rect_callout)
    ax.text(50, 18, "ENGINEERING INNOVATION: Dual-Pointer Header Table with O(1) Tail Appends",
            ha="center", fontsize=10, fontweight="bold", color="#991B1B")
    callout_body = (
        "* Traditional Han et al. FP-Growth: Header table stores only 'head' pointer. Appending new nodes to an item link-chain\n"
        "  requires traversing the entire chain: O(L) time per insertion, degrading to O(N * L^2) on large transaction databases.\n"
        "* skarm Optimization: Header table maintains both 'head' and 'tail' pointers. Appends execute in O(1) constant time:\n"
        "  tail.node_link = new_node; tail = new_node. This eliminated the 3-hour bottleneck on BMS-POS (515,596 transactions),\n"
        "  allowing all 30 evaluation trials to complete smoothly with 100% exact cross-engine itemset agreement."
    )
    ax.text(5, 7.5, callout_body, fontsize=8.2, color="#7F1D1D", linespacing=1.35)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fptree.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {p}")


def draw_ablation():
    """Figure 5 / Section 7.5: Pruning Ablation Study (Bounded vs Exact)."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)

    caps = [50, 100, 250, 500, 1000, 2500, 5000]
    recall = [0.60, 0.60, 4.53, 17.95, 100.0, 100.0, 100.0]
    times = [0.0033, 0.0021, 0.0216, 0.1027, 2.4833, 2.4706, 2.5059]
    itemsets = [16, 16, 121, 479, 2669, 2669, 2669]

    color1 = "#2563EB"
    color2 = "#D95F02"

    ax1.set_xscale("log")
    ax1.plot(caps, recall, marker="o", color=color1, lw=2.2, label="Itemset Recall (%)")
    ax1.set_xlabel("Candidate Budget Cap (max_candidates)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Itemset Recall vs Exact (%)", color=color1, fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-5, 110)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Annotate Degraded Zone vs Exact Zone
    ax1.axvspan(35, 750, color="#FEE2E2", alpha=0.5, label="Degraded Mode (is_degraded_=True)")
    ax1.axvspan(750, 6500, color="#DCFCE7", alpha=0.5, label="Exact Mode (is_degraded_=False)")

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.plot(caps, times, marker="s", color=color2, lw=2.2, linestyle="--", label="Fit Time (seconds)")
    ax2.set_ylabel("Fit Time (seconds, log scale)", color=color2, fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Title & Legend
    plt.title("Pruning Policy Ablation on Chess (support=0.85, 2,669 exact itemsets)\nAccuracy-Speed Trade-off under Safe Candidate Degradation",
              fontsize=10.5, fontweight="bold", pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8.5)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "ablation.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {p}")


if __name__ == "__main__":
    draw_engines()
    draw_architecture()
    draw_pipeline()
    draw_fptree()
    draw_ablation()
    print("All diagrams generated successfully at 300 DPI!")
