import os

path = os.path.join(os.path.dirname(__file__), "..", "paper", "paper.tex")

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Fix Table 1 (Ecosystem table) width so it fits within \textwidth
old_tab1 = r"\begin{tabular}{lllll}"
new_tab1 = r"\begin{tabular}{p{3.2cm}p{3.1cm}p{3.1cm}p{3.1cm}p{3.0cm}}"
if old_tab1 in text:
    text = text.replace(old_tab1, new_tab1)
    print("Fixed Table 1 column widths to fit page margins!")

# 2. Fix Table 2 (Rule metrics) width so conviction column doesn't exceed margin
old_tab2 = r"\begin{tabular}{lll}"
new_tab2 = r"\begin{tabular}{p{2.5cm}p{7.0cm}p{6.0cm}}"
if old_tab2 in text:
    text = text.replace(old_tab2, new_tab2)
    print("Fixed Table 2 column widths to fit page margins!")

# 3. Fix RQ2 Apriori math formula
old_rq2 = r"constrained by combinatorial candidate self-joins ( = L_{k-1} \bowtie L_{k-1})"
new_rq2 = r"constrained by combinatorial candidate self-joins ($C_k = L_{k-1} \bowtie L_{k-1}$)"
if old_rq2 in text:
    text = text.replace(old_rq2, new_rq2)
    print("Fixed RQ2 Apriori candidate join formula!")

# 4. Fix RQ3 speedup and recall numbers
old_rq3 = r"accelerates execution by up to \times$ while recovering meaningful subsets (.60\%$ to .95\%$ recall)."
new_rq3 = r"accelerates execution by up to $15.68\times$ while recovering meaningful subsets ($0.60\%$ to $17.95\%$ recall)."
if old_rq3 in text:
    text = text.replace(old_rq3, new_rq3)
    print("Fixed RQ3 speedup and recall numbers!")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("fix_rq_math.py completed successfully!")
