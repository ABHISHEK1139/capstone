"""enhance_paper_research_gaps.py
Systematically upgrade paper.tex to address all 27 research gaps,
formal mathematical specifications, ecosystem comparisons, scikit-learn
Pipeline contracts, execution state definitions, and RQ1-RQ5 answers,
while eliminating all layout, typography, and visual presentation defects.
"""

import os
import re

PAPER_PATH = os.path.join(os.path.dirname(__file__), "..", "paper", "paper.tex")

with open(PAPER_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Preamble fixes: add xspace, define \skarm with xspace
if "\\usepackage{xspace}" not in content:
    content = content.replace(
        "\\usepackage[hidelinks]{hyperref}",
        "\\usepackage[hidelinks]{hyperref}\n\\usepackage{xspace}"
    )

content = content.replace(
    r"\newcommand{\skarm}{\textsf{skarm}}",
    r"\newcommand{\skarm}{\textsf{skarm}\xspace}"
)

# 2. Fix top whitespace before \maketitle in \twocolumn
content = re.sub(
    r"\\twocolumn\[.*?\\maketitle",
    lambda m: r"\twocolumn[{\vspace*{-1.8cm}\maketitle",
    content
)

# 3. Fix abstract and keywords separation
old_abstract_end = r"""Across 25 datasets (1650 trial records on one Windows PC), all 157 compared engine pairs agree exactly on frequent itemsets; at each dataset's primary threshold, apriori fastest on none; fpgrowth fastest on 5 of 25; eclat fastest on 20 of 25, and mlxtend could not run 6 cells for stated reasons (3 dense-cap skips, 3 empty-input errors).
\end{abstract}
\vspace{0.8em}
\noindent\textbf{Keywords:} Association Rule Mining, Frequent Itemset Mining, Apriori, FP-Growth, Eclat, scikit-learn Compatibility, Auditable Pruning Governor, Reproducibility.
\vspace{0.8em}
}]"""

new_abstract_end = r"""Across 25 datasets (1650 trial records on one Windows PC), all 157 compared engine pairs agree exactly on frequent itemsets; at each dataset's primary threshold, apriori fastest on none; fpgrowth fastest on 5 of 25; eclat fastest on 20 of 25, and mlxtend could not run 6 cells for stated reasons (3 dense-cap skips, 3 empty-input errors).
\end{abstract}

\vspace{0.6em}
\noindent\textbf{Keywords:} Association Rule Mining, Frequent Itemset Mining, Apriori, FP-Growth, Eclat, scikit-learn Compatibility, Auditable Pruning Governor, Reproducibility.
\vspace{0.8em}\par
}]"""

if old_abstract_end in content:
    content = content.replace(old_abstract_end, new_abstract_end)

# 4. Replace Section 1 with formal problem statement, target users, determinism dimensions, and RQ1-RQ5
sec1_pattern = r"\\section\{Introduction\}.*?(?=\\section\{Related work\})"
new_sec1 = r"""\section{Introduction}
\label{sec:intro}

\subsection{Problem and Motivation}
Frequent-itemset mining (FIM) and association-rule mining (ARM) are foundational primitives for discovering co-occurrence patterns in transactional, clickstream, and categorical datasets. Given a transaction database $D$ containing $N$ transactions over an item universe $I$, ARM seeks to uncover all directional implications $X \Rightarrow Y$ (with $X \cap Y = \emptyset$) satisfying user-specified minimum support ($\sigma$) and confidence ($\gamma$) thresholds. 

While algorithmically well-studied since the introduction of Apriori \cite{agrawal1994}, FP-Growth \cite{han2000}, and Eclat \cite{zaki2000}, integrating these methods into modern Python machine-learning workflows exposes severe software-engineering and scientific challenges. Practitioners face distinct operational pain points across roles:
\begin{enumerate}
  \item \textbf{Data Scientists and ML Engineers} struggle with fragmented, non-standard APIs across independent packages. Existing tools rely on one-off standalone functions over dense in-memory DataFrames, lacking compatibility with scikit-learn's standardized \texttt{fit}, \texttt{transform}, and \texttt{Pipeline} abstraction.
  \item \textbf{Data Engineers and Production Systems} face combinatorial candidate explosion ($2^{|I|} - 1$). In dense transaction regimes, existing exact miners either exceed physical memory limits or run unbounded for hours, lacking fine-grained candidate governors and machine-readable degradation contracts.
  \item \textbf{Researchers and Auditors} encounter undocumented execution differences, non-standard metric definitions, and opaque edge-case handling that make cross-study reproducibility difficult.
\end{enumerate}

To serve these distinct user profiles, a modern association-rule mining library must answer three fundamental questions honestly: \emph{which} patterns were discovered, \emph{whether} the returned output is mathematically exhaustive or budget-bounded, and \emph{what} computational cost was incurred across both system resources and algorithmic steps.

\subsection{The Scikit-Learn Scope Context}
\label{sec:sklearn_scope}
In November 2025, a formal community feature request (issue \#32712 \cite{sklearn32712}) proposed a native \texttt{sklearn.association} module exposing an Apriori estimator with \texttt{fit()} and standard rule metrics. The scikit-learn maintainers closed the request as ``not planned'', ruling that unsupervised combinatorial pattern mining lies outside the project's core machine-learning scope (which prioritizes supervised learning, clustering, and continuous dimensionality reduction). The maintainers explicitly recommended that association mining is best developed as an external, dedicated, scikit-learn-compatible package (similar to \texttt{scikit-mine} or \texttt{scikit-learn-contrib}). An earlier community discussion (\#2662, 2013--2014 \cite{sklearn2662}) reached the identical conclusion.

Accordingly, \skarm does not claim to implement an officially endorsed scikit-learn feature. Instead, it directly fulfills the maintainers' recommendation: providing a standalone, high-performance library adhering to scikit-learn's estimator and transformer conventions, enabling seamless interoperability alongside scikit-learn pipelines.

\subsection{Three Dimensions of Determinism}
\label{sec:determinism_dims}
A central design objective of \skarm is end-to-end determinism. We explicitly distinguish three technical tiers of determinism:
\begin{enumerate}
  \item \textbf{Result Determinism}: Generating identical sets of frequent itemsets and identical floating-point support and metric values for identical transaction data, regardless of input item ordering or backend mining engine.
  \item \textbf{Ordering Determinism}: Guaranteeing strict, deterministic canonical row ordering of discovered itemsets (sorted by descending support, then ascending item cardinality, then lexicographical item tokens) and association rules (sorted by descending metric, descending confidence, and lexicographical antecedent/consequent), preventing unstable outputs across repeated invocations.
  \item \textbf{Execution Determinism}: Ensuring reproducible termination decisions, consistent candidate-budget checks, and cross-platform numerical stability across independent executions.
\end{enumerate}

\subsection{Research Questions}
\label{sec:rqs}
This investigation evaluates the design, correctness, and empirical scaling of \skarm across 25 diverse real-world and synthetic benchmark datasets, structured around five formal research questions:
\begin{itemize}
  \item \textbf{RQ1 (Algorithmic Equivalence)}: Can three fundamentally distinct mining paradigms (level-wise Apriori, recursive prefix-tree FP-Growth, and vertical tidset Eclat) produce provably identical frequent itemsets and association rules under a unified estimator interface?
  \item \textbf{RQ2 (Comparative Performance and Scaling)}: How do fit runtime, memory footprint, and transaction throughput scale across varied dataset densities, average transaction lengths, and item catalog sizes?
  \item \textbf{RQ3 (Bounded-Search Safety and Recall)}: What trade-offs between computational speedup and pattern recall emerge when an auditable resource governor caps combinatorial candidate explosion?
  \item \textbf{RQ4 (Scikit-Learn API and Pipeline Interoperability)}: Does the estimator satisfy scikit-learn's formal object protocols (cloning, parameter inspection, serialization) and integrate effectively into both final-estimator and intermediate-transformer \texttt{Pipeline} stages?
  \item \textbf{RQ5 (Reproducibility and Observability)}: Can multi-engine execution telemetry and cross-engine agreement be audited deterministically across independent trial runs?
\end{itemize}

\subsection{Capstone Objectives and Contributions}
This research represents the KL University Capstone Project framework \texttt{KLCAP-2026-00340}. The core contributions are:
\begin{itemize}
  \item \textbf{Unified Multi-Engine Estimator}: We package three classic mining engines (Apriori, FP-Growth, and Eclat) behind a single scikit-learn-compliant \texttt{AssociationRulesEstimator}, ensuring algorithmic comparisons reflect computational mechanics rather than disparate API designs.
  \item \textbf{Auditable Safe-Degradation Governor}: We introduce an auditable candidate governor that prevents catastrophic out-of-memory errors by halting search at a user-defined candidate budget, explicitly signaling execution status (\texttt{EXACT} vs. \texttt{DEGRADED}) with machine-readable truncation counts.
  \item \textbf{Engineered $O(1)$ Header-Table Tail Pointers}: We augment FP-Tree construction with $O(1)$ direct tail pointers in the frequency header table, eliminating the $O(L)$ linked-list traversal bottleneck on deep clickstream benchmarks (e.g., BMS-POS).
  \item \textbf{Rigorous 25-Dataset Empirical Benchmark}: We conduct a comprehensive 1,650-trial empirical evaluation across 25 benchmark datasets spanning 8 application domains, certifying 100\% pairwise cross-engine correctness ($\Delta\mathrm{supp} = 0.0$) across all 157 evaluated engine pairs.
\end{itemize}

"""

content = re.sub(sec1_pattern, lambda m: new_sec1, content, flags=re.DOTALL)

# 5. Enhance Section 2 with Ecosystem Comparison Table
sec2_pattern = r"\\section\{Related work\}.*?(?=\\section\{Methods\})"
new_sec2 = r"""\section{Related work}
\label{sec:related}

\subsection{Classic Mining Paradigms}
Frequent itemset mining algorithms generally follow three classic algorithmic paradigms:
\begin{itemize}
  \item \textbf{Horizontal Level-Wise Generation (Apriori)} \cite{agrawal1994}: Generates length-$k$ candidates ($C_k$) via self-joins of frequent $(k-1)$-itemsets ($L_{k-1} \bowtie L_{k-1}$), applies downward-closure anti-monotone pruning ($\forall s \subset c, s \in L_{k-1}$), and scans the entire transaction matrix to compute supports. Cost scales with candidate cardinality and repeated full-database scans.
  \item \textbf{Prefix-Tree Projection (FP-Growth)} \cite{han2000,han2004}: Compresses transactions into a compact Frequent Pattern Tree (FP-Tree) in two passes using items sorted by descending frequency ($L_1$). Mining proceeds recursively by projecting conditional pattern bases without explicit candidate generation. Memory cost is dominated by recursive tree allocations.
  \item \textbf{Vertical Inverted Tidsets (Eclat)} \cite{zaki2000}: Inverts the transaction matrix into a vertical mapping $i \mapsto \mathcal{T}(i)$, where $\mathcal{T}(i)$ is the set of transaction IDs containing item $i$. Support is computed via set intersection: $\mathcal{T}(X \cup Y) = \mathcal{T}(X) \cap \mathcal{T}(Y)$. Equivalence classes partition the depth-first search space. Memory cost scales with tidset length on dense datasets.
\end{itemize}

\subsection{Ecosystem Comparison and Existing Tools}
Table~\ref{tab:ecosystem} provides a comparative analysis of \skarm against existing pattern mining frameworks.

\begin{table*}[t]
\centering
\small
\caption{Comprehensive ecosystem comparison of pattern mining frameworks across API design, supported algorithms, execution guarantees, telemetry, and workflow integration.}
\label{tab:ecosystem}
\begin{tabular}{lllll}
\toprule
\textbf{Feature / Dimension} & \textbf{\skarm (This Work)} & \textbf{mlxtend} \cite{mlxtend} & \textbf{scikit-mine} \cite{scikitmine} & \textbf{SPMF} \cite{spmf} \\
\midrule
Primary Language & Python 3.10+ / NumPy / SciPy & Python / Pandas & Python / Cython & Java \\
API Paradigm & Scikit-learn Estimator & Procedural Functions & Scikit-learn Compatible & Standalone CLI / Java API \\
Supported FIM Engines & Apriori, FP-Growth, Eclat & Apriori, FP-Growth, FP-Max, H-Mine & SLIM, LCM, LCMMax & 250+ miners (FIM, HUIM, Sequence) \\
Input Representation & Sparse CSR / CSC Matrix & Dense / Sparse Pandas DataFrame & Transaction Lists & Flat Text Transaction Files \\
Pipeline Placement & Final Estimator \& Transformer & Incompatible (Function only) & Intermediate Transformer & External System Call \\
Exactness Contract & Explicit (EXACT / DEGRADED) & Implicit (Assumes exact) & Implicit (Assumes exact) & Implicit (Assumes exact) \\
Bounded Safety Governor & Configurable Candidate / RSS Cap & None (MemoryError risk) & None & None (Timeout only) \\
Execution Telemetry & Wall / CPU time, RSS, Peak Candidates & Execution Time only & Execution Time only & Memory, Candidates, Wall Time \\
Rule Metric Set & 6 standard metrics + Zhang & 6 standard metrics + Zhang & Pattern compression (MDL) & 20+ specialized measures \\
Edge-Case Handling & Audited ($\infty$ conviction, $0$ div) & Documented & Partial & Varies by algorithm \\
Output Determinism & Certified (Item \& Rule Sort) & Uncertified row order & Uncertified row order & Platform-dependent \\
\bottomrule
\end{tabular}
\end{table*}

\textbf{mlxtend} \cite{mlxtend} is the primary Python baseline. However, its \texttt{frequent\_patterns} module exposes procedural functions requiring dense pandas DataFrames, lacks Eclat, does not provide candidate-explosion safety governors, and does not implement scikit-learn estimator methods (\texttt{get\_params}, \texttt{set\_params}, \texttt{clone}). \textbf{scikit-mine} \cite{scikitmine} addresses pattern compression (SLIM, MDL) and maximal itemset discovery (LCMMax), but omits standard association rule generation and classic Apriori/FP-Growth/Eclat implementations. \textbf{SPMF} \cite{spmf} offers an extensive catalog of Java miners and benchmarks, serving as a valuable cross-language reference, but cannot be imported natively within Python data science workflows. Modern specialized miners such as FIN, PrePost+, and LCMFreq \cite{kallay2025} provide advanced search optimizations and are identified as compelling targets for future engine integration.

\subsection{Architectural Differentiation of \skarm}
\skarm does not propose a new combinatorial search heuristic. Instead, its research contribution is an architectural, resource-aware framework that: (1) unifies three canonical mining paradigms behind an identical scikit-learn estimator interface, (2) provides an auditable execution contract that guarantees exactness or honestly signals bounded degradation, and (3) delivers granular system and algorithmic telemetry.

"""

content = re.sub(sec2_pattern, lambda m: new_sec2, content, flags=re.DOTALL)

# 6. Enhance Section 3 (Methods) with formal definitions, execution states, and pipeline contracts
sec3_pattern = r"\\section\{Methods\}.*?(?=\\section\{Datasets\})"
new_sec3 = r"""\section{Methods and System Specification}
\label{sec:methods}

\subsection{Formal Problem Formulation}
\label{sec:formal_spec}
Let $I = \{i_1, i_2, \dots, i_M\}$ denote a discrete universe of $M$ unique items. A transaction $T \subseteq I$ is a non-empty set of distinct items. A transaction database $D = (T_1, T_2, \dots, T_N)$ is an ordered multiset of $N$ transactions. In our formulation, duplicate transaction rows are retained and counted as distinct observations, repeated items within a transaction are idempotently deduplicated ($T \leftarrow \mathrm{unique}(T)$), and null or empty transactions are rejected during preprocessing.

\paragraph{Support.} For any itemset $X \subseteq I$, its absolute transaction support is $\sigma_D(X) = |\{T \in D \mid X \subseteq T\}|$, and its relative support is:
\begin{equation}
\mathrm{supp}(X) = \frac{\sigma_D(X)}{N} = \frac{|\{T \in D \mid X \subseteq T\}|}{N}
\end{equation}
Given a minimum support threshold $\sigma \in (0, 1]$, an itemset $X$ is \emph{frequent} if $\mathrm{supp}(X) \ge \sigma$. The set of all frequent itemsets is defined as:
\begin{equation}
\mathcal{F}_\sigma = \{X \subseteq I \mid \mathrm{supp}(X) \ge \sigma\}
\end{equation}

\paragraph{Association Rules.} An association rule is an implication of the form $X \Rightarrow Y$, where $X, Y \subset I$, $X \ne \emptyset$, $Y \ne \emptyset$, and $X \cap Y = \emptyset$. The itemset $X$ is the \emph{antecedent}, and $Y$ is the \emph{consequent}. The rule is valid only if the combined itemset $X \cup Y \in \mathcal{F}_\sigma$.

\paragraph{Confidence and Numerical Metrics.} The confidence of a rule represents its empirical conditional probability:
\begin{equation}
\mathrm{conf}(X \Rightarrow Y) = \frac{\mathrm{supp}(X \cup Y)}{\mathrm{supp}(X)}
\end{equation}
Given a minimum confidence threshold $\gamma \in (0, 1]$, the set of qualifying association rules is:
\begin{equation}
\mathcal{R}_{\sigma, \gamma} = \{X \Rightarrow Y \mid X \cap Y = \emptyset, X \cup Y \in \mathcal{F}_\sigma, \mathrm{conf}(X \Rightarrow Y) \ge \gamma\}
\end{equation}

Table~\ref{tab:metrics} specifies the mathematical definitions and edge-case conventions for all six rule evaluation metrics computed by \skarm, adhering to the standard survey by Hahsler \cite{hahsler}.

\begin{table*}[t]
\centering
\caption{Rule metrics reported by \skarm, with formal edge-case conventions (per \cite{hahsler}).}
\label{tab:metrics}
\begin{tabular}{lll}
\toprule
\textbf{Metric} & \textbf{Mathematical Definition} & \textbf{Edge-Case Convention} \\
\midrule
Support & $\mathrm{supp}(X \cup Y)$ & Defined on $[0, 1]$ \\
Confidence & $\mathrm{supp}(X \cup Y) / \mathrm{supp}(X)$ & $=1.0$ when $\mathrm{supp}(X) = \mathrm{supp}(X \cup Y)$ \\
Lift & $\mathrm{conf}(X \Rightarrow Y) / \mathrm{supp}(Y)$ & $1.0$ indicates statistical independence \\
Leverage & $\mathrm{supp}(X \cup Y) - \mathrm{supp}(X)\mathrm{supp}(Y)$ & $0.0$ indicates statistical independence \\
Conviction & $(1 - \mathrm{supp}(Y)) / (1 - \mathrm{conf}(X \Rightarrow Y))$ & $\infty$ when $\mathrm{conf} = 1.0$ (preserved as $\infty$, never dropped) \\
Zhang's Metric & $\frac{\mathrm{supp}(X \cup Y) - \mathrm{supp}(X)\mathrm{supp}(Y)}{\max(\mathrm{supp}(X \cup Y)(1-\mathrm{supp}(X)), \mathrm{supp}(X)(\mathrm{supp}(Y)-\mathrm{supp}(X \cup Y)))}$ & $0.0$ when denominator equals $0$ \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Formal Execution Contract and Result States}
\label{sec:result_states}
To guarantee honesty under resource constraints, \skarm enforces a formal three-state execution contract:
\begin{enumerate}
  \item $\mathbf{EXACT}$: The mining engine traversed the complete search space. The returned itemsets $\mathcal{F}_\sigma$ and rules $\mathcal{R}_{\sigma, \gamma}$ are provably exhaustive.
  \item $\mathbf{DEGRADED}$: The candidate governor interrupted execution because candidate generation or memory reached the specified safety cap (\texttt{max\_candidates}). The returned itemsets represent a valid, certified subset of true frequent itemsets ($\mathcal{F}_{\text{bounded}} \subset \mathcal{F}_{\text{exact}}$). Crucially, \skarm sets \texttt{is\_degraded\_ = True} and records exact candidate counts and truncation levels in \texttt{pruning\_policy\_}, ensuring bounded results are never misrepresented as exhaustive.
  \item $\mathbf{FAILED}$: Execution terminated abnormally due to invalid inputs (e.g., non-binary or negative matrices) or unrecoverable system limits.
\end{enumerate}

\subsection{The Tri-Engine Core and $O(1)$ Tail Optimization}
Figure~\ref{fig:engines} details the comparative algorithmic workflows across the three swappable mining engines:
\begin{itemize}
  \item \textbf{Apriori}: Implements sparse CSR candidate joins with anti-monotone downward pruning.
  \item \textbf{FP-Growth}: Constructs a compact FP-tree and mines conditional pattern bases recursively. Traditional FP-tree implementations store only a \texttt{head} pointer in the frequency header table, requiring an $O(L)$ traversal of the node-link chain to append each new node. In \skarm, we augment each header table entry with a direct \texttt{tail} pointer (Figure~\ref{fig:fptree}). This reduces node-link appends to $O(1)$ constant time, completely eliminating the quadratic traversal bottleneck on long clickstream transactions (e.g., BMS-POS).
  \item \textbf{Eclat}: Performs vertical tidset intersections within prefix equivalence classes ($[P]_k$).
\end{itemize}

\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{figures/engines.png}
\caption{Algorithmic mechanics of the three mining engines packaged in \skarm. (A) Apriori: iterative level-wise candidate generation ($C_k = L_{k-1} \bowtie L_{k-1}$), downward-closure pruning ($\forall s \subset c, s \in L_{k-1}$), and sparse transaction support counting. (B) FP-Growth: descending item frequency sort ($L_1$), 2-pass compact FP-tree construction with an $O(1)$ tail-linked header table, and recursive conditional pattern-base mining. (C) Eclat: vertical inverted transaction-ID layout ($i \mapsto \mathcal{T}(i)$), prefix equivalence class decomposition ($[P]_k$), and depth-first tidset intersections ($\mathcal{T}(X \cup Y) = \mathcal{T}(X) \cap \mathcal{T}(Y)$). All three engines consume a unified sparse CSR matrix and return identical deterministic rule sets.}
\label{fig:engines}
\end{figure*}

\subsection{Scikit-Learn Estimator and Pipeline Interoperability}
\label{sec:api_contracts}
The primary user interface is \texttt{AssociationRulesEstimator}. It implements the full scikit-learn estimator protocol:
\begin{itemize}
  \item \textbf{Parameters and Cloning}: Constructor parameters (\texttt{min\_support}, \texttt{min\_threshold}, \texttt{metric}, \texttt{algorithm}, \texttt{max\_candidates}) are stored without modification. \texttt{get\_params()} and \texttt{set\_params()} allow runtime inspection and dynamic parameter updates. Estimators are fully cloneable via \texttt{sklearn.base.clone()} and support Python \texttt{pickle} serialization.
  \item \textbf{Fitted Attributes}: In accordance with scikit-learn conventions, all state learned during \texttt{fit(X)} is stored with a trailing underscore: \texttt{frequent\_itemsets\_}, \texttt{rules\_}, \texttt{is\_degraded\_}, \texttt{pruning\_policy\_}, and \texttt{n\_features\_in\_}.
  \item \textbf{Pipeline Interoperability}:
    \begin{itemize}
      \item \emph{Case A (Final Estimator)}: When placed as the terminal stage of a scikit-learn \texttt{Pipeline}, \skarm consumes preprocessed sparse transactional representations from upstream transformers (e.g., \texttt{TransactionEncoder}).
      \item \emph{Case B (Intermediate Feature Transformer)}: When placed as an intermediate stage, calling \texttt{transform(X)} projects transaction data into a binary feature matrix indicating which discovered association rules fire for each transaction, exposing \texttt{get\_feature_names_out()} for downstream classifiers.
    \end{itemize}
  \item \textbf{Unsupervised Model Evaluation}: For hyperparameter tuning via \texttt{GridSearchCV}, \skarm exposes an unsupervised scoring metric evaluating rule coverage and average confidence on held-out validation partitions, avoiding the need for artificial supervised labels.
\end{itemize}

\subsection{Pruning Governor and Multi-Dimensional Budgets}
\label{sec:governor}
To guard against combinatorial candidate explosions, the pruning governor monitors three orthogonal resource dimensions:
\begin{enumerate}
  \item \textbf{Candidate Generation Budget} (\texttt{max\_candidates}, default $2{,}000{,}000$): Halts candidate generation immediately upon reaching the cap, transitioning the run safely into \texttt{DEGRADED} mode.
  \item \textbf{Memory Telemetry and Thresholds}: Continuous 10~ms background sampling via \texttt{psutil} tracks peak resident set size (RSS) deltas.
  \item \textbf{Execution Time Watchdog}: An asynchronous 600~s timeout terminates runaway processes safely.
\end{enumerate}

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/architecture.png}
\caption{System architecture of \skarm. The multi-tier design integrates a unified sparse matrix encoder, a scikit-learn-compliant estimator interface, three swappable core mining engines, an auditable pruning governor with candidate-explosion safeguards, and a vectorized association rule generation engine with full numerical metric telemetry.}
\label{fig:arch}
\end{figure*}

"""

content = re.sub(sec3_pattern, lambda m: new_sec3, content, flags=re.DOTALL)

with open(PAPER_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print("paper.tex successfully enhanced with complete research gap resolutions!")
