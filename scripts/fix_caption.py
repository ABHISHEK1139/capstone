import os

path = os.path.join(os.path.dirname(__file__), "..", "paper", "paper.tex")

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

bad = r"\caption{FP-tree architecture with frequency header table and the (1)$ tail-pointer optimization. Standard FP-tree implementations store only a \texttt{head} pointer, requiring an (L)$ linked-list traversal to append new nodes during tree insertion. \skarm augments each header entry with a direct \texttt{tail} pointer, reducing node-link appends to (1)$ constant time. This optimization eliminated the sequential traversal bottleneck on deep clickstream benchmarks (such as BMS-POS), enabling all 30 evaluation trials to complete smoothly with zero correctness deviations.}"

good = r"\caption{FP-tree architecture with frequency header table and the $O(1)$ tail-pointer optimization. Standard FP-tree implementations store only a \texttt{head} pointer, requiring an $O(L)$ linked-list traversal to append new nodes during tree insertion. \skarm augments each header entry with a direct \texttt{tail} pointer, reducing node-link appends to $O(1)$ constant time. This optimization eliminated the sequential traversal bottleneck on deep clickstream benchmarks (such as BMS-POS), enabling all 30 evaluation trials to complete smoothly with zero correctness deviations.}"

if bad in text:
    text = text.replace(bad, good)
    print("Found and replaced bad caption!")
else:
    print("Bad caption string not found directly, performing line-by-line fix")
    lines = text.splitlines(True)
    for i, line in enumerate(lines):
        if "tail-pointer optimization" in line:
            lines[i] = good + "\n"
            print(f"Fixed line {i+1}")
    text = "".join(lines)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Caption fix script finished.")
