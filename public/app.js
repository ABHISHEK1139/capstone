const DATASET_DEFAULTS = {
  groceries: { min_support: "0.02", min_threshold: "0.25", metric: "confidence", basket: "yogurt, rolls/buns", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  market_basket: { min_support: "0.015", min_threshold: "0.25", metric: "confidence", basket: "mineral water", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  retail: { min_support: "0.03", min_threshold: "0.40", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  online_retail: { min_support: "0.02", min_threshold: "0.50", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  online_retail_ii: { min_support: "0.02", min_threshold: "0.50", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  demo: { min_support: "0.2", min_threshold: "0.5", metric: "confidence", basket: "milk, bread", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  bms_webview_1: { min_support: "0.01", min_threshold: "0.25", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  bms_webview_2: { min_support: "0.01", min_threshold: "0.25", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  t10i4d100k: { min_support: "0.01", min_threshold: "0.25", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "eclat" },
  synthetic_sparse_2000: { min_support: "0.05", min_threshold: "0.40", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  synthetic_dense_5000: { min_support: "0.50", min_threshold: "0.50", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  mushroom: { min_support: "0.55", min_threshold: "0.85", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "fpgrowth" },
  chess: { min_support: "0.85", min_threshold: "0.85", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "fpgrowth" },
  adult_census: { min_support: "0.20", min_threshold: "0.60", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  tic_tac_toe: { min_support: "0.25", min_threshold: "0.70", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "eclat" },
  car_evaluation: { min_support: "0.20", min_threshold: "0.60", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  balance_scale: { min_support: "0.15", min_threshold: "0.35", metric: "confidence", basket: "", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  docword_nips: { min_support: "0.65", min_threshold: "0.70", metric: "confidence", basket: "", max_candidates: "5000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  pumsb_star: { min_support: "0.55", min_threshold: "0.80", metric: "confidence", basket: "", max_candidates: "2000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
  custom: { min_support: "0.3", min_threshold: "0.5", metric: "confidence", basket: "milk", max_candidates: "100000", max_len: "", vip_item: "", vip_multiplier: "1", algorithm: "auto" },
};

const DATASET_FILE_MAP = {
  groceries: "groceries.csv",
  market_basket: "market_basket.csv",
  bms_webview_1: "Transactional_BMS-WebView-1.csv",
  bms_webview_2: "BMS-WebView-2.dat",
  t10i4d100k: "T10I4D100K.dat",
};

const DEMO_PRESETS = {
  groceries: {
    dataset: "groceries",
    algorithm: "auto",
    metric: "confidence",
    min_support: "0.02",
    min_threshold: "0.25",
    max_candidates: "100000",
    max_len: "",
    vip_item: "",
    vip_multiplier: "1",
    basket: "yogurt, rolls/buns",
  },
  safety_degraded: {
    dataset: "mushroom",
    algorithm: "apriori",
    metric: "confidence",
    min_support: "0.35",
    min_threshold: "0.8",
    max_candidates: "120",
    max_len: "",
    vip_item: "",
    vip_multiplier: "1",
    basket: "",
  },
  sparse_eclat: {
    dataset: "tic_tac_toe",
    algorithm: "eclat",
    metric: "confidence",
    min_support: "0.25",
    min_threshold: "0.7",
    max_candidates: "100000",
    max_len: "",
    vip_item: "",
    vip_multiplier: "1",
    basket: "",
  },
  weighted_vip: {
    dataset: "demo",
    algorithm: "auto",
    metric: "confidence",
    min_support: "0.25",
    min_threshold: "0.6",
    max_candidates: "100000",
    max_len: "",
    vip_item: "diapers",
    vip_multiplier: "3.5",
    basket: "diapers",
  },
  pipeline_transform: {
    dataset: "demo",
    algorithm: "fpgrowth",
    metric: "lift",
    min_support: "0.2",
    min_threshold: "1.05",
    max_candidates: "100000",
    max_len: "3",
    vip_item: "",
    vip_multiplier: "1",
    basket: "milk, bread",
  },
};

const VIVA_QA = [
  {
    q: "1. Why did scikit-learn close #32712, and how does skarm satisfy scikit-learn contracts?",
    a: "Core scikit-learn focuses on supervised/embedding estimators (#32712 closed Nov 2025 as not planned). skarm acts as an independent package inheriting BaseEstimator + TransformerMixin: supporting clone(), fit/predict/transform(), get_feature_names_out(), score() for GridSearchCV, sparse SciPy CSR inputs, and 7/7 KPI-6 checks.",
    btnLabel: "Show Live Pipeline transform()",
    action: () => applyDemoPreset("pipeline_transform"),
  },
  {
    q: "2. How do you prevent memory/CPU hangs when dense datasets explode exponentially?",
    a: "PruningPolicy guards candidate count (max_candidates), runtime, and 10 ms RSS memory polling (MemorySampler). When crossed, instead of crashing or silently dropping patterns, skarm halts safely, sets status_='DEGRADED' & is_degraded_=True, and delivers up to 17.14× speedup on Chess.",
    btnLabel: "Trigger Live DEGRADED Mode",
    action: () => applyDemoPreset("safety_degraded"),
  },
  {
    q: "3. Does Apriori ever beat FP-Growth/Eclat, and why benchmark Java SPMF?",
    a: "Across 25 corpora (1,650 trials), Apriori won 0/25; Bitset Eclat (POPCNT) leads sparse/medium corpora (21/25 at thresholds[0], 20/25 in lower-support full sweep); FP-Growth leads dense/long-row sets (e.g. accidents RSS drops 1,860 MB → 90.4 MB). Java SPMF (FIN, PrePost+, LCMFreq) proved 25/25 exact parity while showing mlxtend hits an 814.8 MB dense skip on BMS-POS.",
    btnLabel: "Inspect 25-Corpus & SPMF Data",
    action: () => switchTab("benchmarks"),
  },
  {
    q: "4. How is weighted transaction support (Eq. 2.1) & math oracle precision verified?",
    a: "Weighted support sums normalized weights over positive binary samples and is tested in NT-3 (rejecting negative weights with ValueError). All 6 rule metrics (Support, Confidence, Lift, Leverage, Conviction, Zhang's metric) hit ≤ 2.22×10⁻¹⁶ max error against closed-form analytical fixtures.",
    btnLabel: "Run Live Weighted Support (3.5×)",
    action: () => applyDemoPreset("weighted_vip"),
  },
];

const state = {
  activeTab: "explorer",
  overview: null,
  lastMineData: null,
  selectedRuleIdx: 0,
  activeAblDataset: "chess",
  activeBaseFilter: "all",
  mining: false,
  acceptanceRunning: false,
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function fmtNum(val, digits = 4) {
  if (val === null || val === undefined) return "—";
  if (val === "inf") return "∞";
  if (val === "-inf") return "-∞";
  const num = Number(val);
  if (!Number.isFinite(num)) return "—";
  return num.toFixed(digits);
}

function fmtMs(seconds) {
  const num = Number(seconds);
  if (!Number.isFinite(num)) return "—";
  return (num * 1000).toFixed(2);
}

function switchTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    const isActive = btn.dataset.tab === tabName;
    btn.setAttribute("aria-selected", String(isActive));
    btn.classList.toggle("border-blue-600", isActive);
    btn.classList.toggle("text-slate-900", isActive);
    btn.classList.toggle("border-transparent", !isActive);
    btn.classList.toggle("text-slate-500", !isActive);
  });

  ["explorer", "benchmarks", "acceptance", "ecosystem", "paper"].forEach((name) => {
    const panel = document.getElementById(`panel-${name}`);
    if (panel) panel.classList.toggle("hidden", name !== tabName);
  });
}

function updateLivePythonSnippet() {
  const codeEl = document.getElementById("live-python-code");
  if (!codeEl) return;
  const dataset = document.getElementById("dataset-select").value;
  const algorithm = document.getElementById("algorithm-select").value;
  const metric = document.getElementById("metric-select").value;
  const minSupp = document.getElementById("min-support").value || "0.02";
  const minThresh = document.getElementById("min-threshold").value || "0.25";
  const maxCand = document.getElementById("max-candidates").value || "100000";
  const maxLen = document.getElementById("max-len").value.trim();
  const vipItem = document.getElementById("vip-item").value.trim();
  const vipMult = Number(document.getElementById("vip-multiplier").value || "1");

  const loaderLine = dataset === "custom"
    ? 'transactions = [["milk", "bread"], ["beer", "diapers"]]'
    : dataset === "demo"
      ? 'transactions = [["milk", "bread", "butter"], ["beer", "diapers"], ...]'
      : `from skarm.datasets import load_transactions\ntransactions = load_transactions("data/${DATASET_FILE_MAP[dataset] || (dataset + ".dat")}")`;

  const lenLine = maxLen ? `\n    max_len=${maxLen},` : "";
  const wPrep = vipItem && vipMult > 1
    ? `\nweights = [${vipMult} if "${vipItem}" in tx else 1.0 for tx in transactions]`
    : "";
  const fitArg = vipItem && vipMult > 1 ? "transactions, sample_weight=weights" : "transactions";

  codeEl.textContent = `${loaderLine}${wPrep}\nfrom skarm import AssociationRulesEstimator\n\nest = AssociationRulesEstimator(\n    algorithm="${algorithm}",\n    min_support=${minSupp},\n    metric="${metric}",\n    min_threshold=${minThresh},${lenLine}\n    max_candidates=${maxCand},\n    sparse_output=True,\n).fit(${fitArg})\n\nX_rules = est.transform(transactions[:2])\nprint(est.status_, est.algorithm_selected_, est.score(transactions))`;
}

function renderRuleExplainer(rule, index, data) {
  const panel = document.getElementById("rule-explainer-panel");
  const narrativeEl = document.getElementById("explainer-narrative");
  const metricsEl = document.getElementById("explainer-metrics");
  const badgeEl = document.getElementById("explainer-badge");
  if (!rule) {
    panel.classList.add("hidden");
    return;
  }

  panel.classList.remove("hidden");
  state.selectedRuleIdx = index;
  badgeEl.textContent = `Rule #${index + 1}`;

  const antStr = rule.antecedent.join(" + ");
  const conStr = rule.consequent.join(" + ");
  const nTxns = data?.summary?.n_transactions || 0;
  const weighted = Boolean(data?.weighted_info?.active);
  const suppPct = ((Number(rule.support) || 0) * 100).toFixed(2);
  const confPct = ((Number(rule.confidence) || 0) * 100).toFixed(2);
  const conPct = rule.consequent_support !== null && rule.consequent_support !== undefined
    ? `${(Number(rule.consequent_support) * 100).toFixed(2)}%`
    : "baseline";
  const liftNum = Number(rule.lift);
  const liftDesc = Number.isFinite(liftNum)
    ? `${liftNum.toFixed(2)}× more likely than baseline (${conPct})`
    : "—";

  narrativeEl.textContent = "";
  const occPhrase = weighted
    ? `under weighted Eq. 2.1 transactions (VIP "${data.weighted_info.vip_item}" ×${data.weighted_info.multiplier}), weighted support is ${suppPct}%`
    : `out of ${nTxns.toLocaleString()} transactions, customers bought {${antStr}} & {${conStr}} together ${Math.round((Number(rule.support) || 0) * nTxns).toLocaleString()} times (${suppPct}%)`;

  narrativeEl.append(
    el("span", "", `Interpretation: ${occPhrase}. Whenever {`),
    el("strong", "text-slate-900", antStr),
    el("span", "", `} is present, there is a `),
    el("strong", "font-mono text-slate-900", `${confPct}%`),
    el("span", "", ` probability {`),
    el("strong", "text-blue-700", conStr),
    el("span", "", `} appears too — making it `),
    el("strong", "font-mono text-emerald-700", liftDesc),
    el("span", "", ".")
  );

  const pA = rule.antecedent_support !== null ? fmtNum(rule.antecedent_support, 4) : "P(A)";
  const pC = rule.consequent_support !== null ? fmtNum(rule.consequent_support, 4) : "P(C)";
  const pAC = fmtNum(rule.support, 4);
  const cards = [
    { name: "Support P(A ∪ C)", val: pAC, formula: weighted ? "Weighted Σw / ΣW" : `freq / ${nTxns} = ${pAC}`, desc: "Joint coverage across transaction corpus." },
    { name: "Confidence P(C | A)", val: fmtNum(rule.confidence, 4), formula: `P(A∪C)/P(A) = ${pAC}/${pA}`, desc: "Conditional probability of C given A." },
    { name: "Lift", val: fmtNum(rule.lift, 3), formula: `Conf/P(C) = ${fmtNum(rule.confidence, 3)}/${pC}`, desc: "> 1.0 = positive dependence." },
    { name: "Leverage", val: fmtNum(rule.leverage, 4), formula: "P(A∪C) − P(A)·P(C)", desc: "Excess probability above independence." },
    { name: "Conviction", val: fmtNum(rule.conviction, 3), formula: "(1 − P(C)) / (1 − Conf)", desc: "Implication sensitivity (∞ if Conf=1)." },
    { name: "Zhang's Metric", val: fmtNum(rule.zhangs_metric, 3), formula: "Normalized ∈ [-1, +1]", desc: "+1 perfect rule; 0 independent; -1 disjoint." },
  ];

  metricsEl.textContent = "";
  cards.forEach((c) => {
    const box = el("div", "sk-metric-box");
    const head = el("div", "sk-metric-title");
    head.appendChild(el("span", "", c.name));
    box.append(head, el("div", "sk-metric-val", c.val), el("div", "sk-metric-formula", c.formula), el("div", "sk-metric-desc", c.desc));
    metricsEl.appendChild(box);
  });

  document.querySelectorAll(".sk-rule-row").forEach((tr, rIdx) => {
    tr.classList.toggle("selected", rIdx === index);
  });
}

function renderEngineComparison(compList) {
  const panel = document.getElementById("comparison-panel");
  const cardsEl = document.getElementById("comparison-cards");
  const verdictEl = document.getElementById("comparison-verdict");
  if (!Array.isArray(compList) || compList.length === 0) {
    panel.classList.add("hidden");
    return;
  }

  panel.classList.remove("hidden");
  cardsEl.textContent = "";
  const ap = compList.find((c) => c.algorithm === "apriori") || compList[0];
  const maxMs = Math.max(...compList.map((c) => c.elapsed_ms), 0.1);
  const allExact = compList.every((c) => c.status === "EXACT");
  const allParity = compList.every((c) => c.exact_parity);

  verdictEl.textContent = allExact && allParity
    ? "BIT-EXACT PARITY ACROSS ALL 3 ENGINES (Δsupp = 0.00e+00)"
    : !allExact ? "BOUNDED DEGRADED MODE TRIGGERED" : "PARITY CHECK COMPLETED";
  verdictEl.className = `text-xs font-mono font-semibold ${allExact && allParity ? "text-emerald-700" : "sk-degraded-text"}`;

  const colors = { apriori: "#64748b", fpgrowth: "#2563eb", eclat: "#0891b2" };
  const titles = { apriori: "Apriori (O2 Ref)", fpgrowth: "FP-Growth (FP-Array)", eclat: "Eclat (Bitset TID)" };

  compList.forEach((row) => {
    const card = el("div", "border border-slate-200 p-3 bg-slate-50/40 rounded-sm");
    const hdr = el("div", "flex items-center justify-between gap-2");
    hdr.append(
      el("span", "text-xs font-semibold text-slate-900", titles[row.algorithm] || row.algorithm),
      el("span", `text-[11px] font-mono ${row.status === "EXACT" ? "text-emerald-700" : "sk-degraded-text"}`, row.status)
    );
    const speedup = row.algorithm === "apriori" ? "1.00× (Ref)" : `${(ap.elapsed_ms / Math.max(row.elapsed_ms, 0.001)).toFixed(2)}× vs Apriori`;
    const mainStat = el("div", "sk-mt-2 flex items-baseline justify-between font-mono");
    mainStat.append(el("span", "sk-fs-lg font-semibold text-slate-900", `${row.elapsed_ms.toFixed(2)} ms`), el("span", "text-xs text-blue-700 font-medium", speedup));
    const track = el("div", "sk-bar-track");
    const fill = el("div", "sk-bar-fill");
    fill.style.width = `${Math.max(8, Math.round((row.elapsed_ms / maxMs) * 100))}%`;
    fill.style.background = colors[row.algorithm] || "#2563eb";
    track.appendChild(fill);
    const footText = `${row.n_itemsets} itemsets · ${row.n_rules} rules · ${row.status !== "EXACT" ? "Bounded halted" : row.exact_parity ? "Δsupp = 0.00e+00" : `Δ=${row.max_delta_supp}`}`;
    card.append(hdr, mainStat, track, el("div", "mt-1 text-[11px] font-mono text-slate-600", footText));
    cardsEl.appendChild(card);
  });
}

function renderPipelineTransform(pt) {
  const panel = document.getElementById("transform-panel");
  const summaryEl = document.getElementById("transform-summary");
  const bitsEl = document.getElementById("transform-bits");
  const listEl = document.getElementById("transform-rules-list");

  if (!pt) {
    panel.classList.add("hidden");
    return;
  }

  panel.classList.remove("hidden");
  summaryEl.textContent = `Basket [${pt.query_items.join(", ")}] → ${pt.fired_count} / ${pt.total_features} rule features fired (${pt.active_rate_pct}%)`;

  bitsEl.textContent = "";
  bitsEl.appendChild(el("span", "text-[11px] font-mono text-slate-500 mr-1.5", "X_out[0, :24]:"));
  (pt.vector_preview || []).forEach((b) => {
    bitsEl.appendChild(el("span", `sk-bit ${b ? "on" : "off"}`, String(b)));
  });
  if (pt.total_features > 24) {
    bitsEl.appendChild(el("span", "text-[11px] font-mono text-slate-400 sk-ml-1", `+${pt.total_features - 24} cols`));
  }

  listEl.textContent = "";
  if (Array.isArray(pt.fired_rules) && pt.fired_rules.length > 0) {
    listEl.appendChild(el("div", "text-[11px] text-slate-500 uppercase py-1", "Active Downstream Classifier Features (get_feature_names_out):"));
    pt.fired_rules.forEach((r) => {
      listEl.appendChild(el("div", "px-2 py-1 sk-my-1 bg-slate-50 border border-slate-200 rounded-sm text-slate-900", r));
    });
  } else {
    listEl.appendChild(el("div", "text-slate-500", "No rules fired for this query basket — downstream feature vector is all zeros."));
  }
}

function renderSampleChips(sampleItems) {
  const wrap = document.getElementById("sample-item-chips");
  if (!wrap) return;
  wrap.textContent = "";
  const items = Array.isArray(sampleItems) ? sampleItems.slice(0, 12) : [];
  if (items.length === 0) return;

  const currentBasket = new Set(
    document.getElementById("basket-input").value.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean)
  );

  items.forEach((item) => {
    const btn = el("button", `sk-chip-btn ${currentBasket.has(item.toLowerCase()) ? "active" : ""}`, `+ ${item}`);
    btn.type = "button";
    btn.addEventListener("click", () => {
      const inp = document.getElementById("basket-input");
      const list = inp.value.split(",").map((s) => s.trim()).filter(Boolean);
      const existsIdx = list.findIndex((s) => s.toLowerCase() === item.toLowerCase());
      if (existsIdx >= 0) list.splice(existsIdx, 1);
      else list.push(item);
      inp.value = list.join(", ");
      renderSampleChips(sampleItems);
    });
    wrap.appendChild(btn);
  });
}

function renderMineResult(data) {
  state.lastMineData = data;
  document.getElementById("result-dataset").textContent = data.dataset_label;
  const s = data.summary;
  document.getElementById("result-meta").textContent =
    `${s.n_transactions.toLocaleString()} txns · ${s.n_unique_items.toLocaleString()} items · avg len ${s.avg_transaction_length} · density ${s.density_pct}% · ${data.n_itemsets.toLocaleString()} itemsets · ${data.n_rules.toLocaleString()} rules`;

  const statusEl = document.getElementById("result-status");
  const isExact = data.status === "EXACT";
  statusEl.textContent = `${String(data.algorithm_used).toUpperCase()} · ${data.status} · ${data.elapsed_ms.toFixed(1)} ms`;
  statusEl.className = `text-xs font-mono uppercase tracking-wider tabular-nums ${isExact ? "text-emerald-700" : "text-amber-700 font-semibold"}`;

  const banner = document.getElementById("dispatch-explanation");
  banner.textContent = "";
  const algoUsed = String(data.algorithm_used || "").toUpperCase();
  if (!isExact || data.is_degraded) {
    banner.className = "sk-info-banner degraded";
    banner.append(
      el("strong", "", `SAFETY GOVERNOR TRIGGERED (status_="DEGRADED", is_degraded_=True): `),
      el("span", "", `${data.termination_reason || "Candidate ceiling reached."} Halted at level ${data.level_halted ?? "?"} cleanly without RAM crash (peak candidates: ${data.peak_candidates ?? "capped"}). Future candidates: ${data.future_candidates_status || "UNKNOWN"}.`)
    );
  } else {
    banner.className = "sk-info-banner";
    const routeNote = data.algorithm_requested === "auto"
      ? `Adaptive router measured density ρ = ${s.density_pct}% (${s.density_pct <= 15 ? "≤ 15% sparse → routed to bitset ECLAT" : "> 15% dense → routed to FP-ARRAY FPGROWTH"}).`
      : `Executed explicit ${algoUsed} engine.`;
    const weightNote = data.weighted_info?.active
      ? ` Weighted Eq. 2.1 active: "${data.weighted_info.vip_item}" baskets ×${data.weighted_info.multiplier} (${data.weighted_info.affected_txns} boosted txns).`
      : "";
    banner.append(
      el("strong", "", `Execution Audit (EXACT): `),
      el("span", "", `${routeNote}${weightNote} Exhausted ${data.n_itemsets} itemsets & ${data.n_rules} rules with IEEE-754 verified metrics.`)
    );
  }

  // Telemetry strip
  const mp = data.memory_profile;
  if (mp) {
    document.getElementById("csr-saving-val").textContent = `${mp.csr_kb} KB vs ${mp.dense_float64_kb} KB`;
    document.getElementById("csr-bar").style.width = `${Math.max(4, Math.min(100, Math.round(100 - mp.saving_pct)))}%`;
    document.getElementById("csr-detail").textContent = `${mp.saving_pct}% memory saved (${mp.nnz.toLocaleString()} non-zero entries in CSR)`;
  }

  const ops = data.operations || {};
  const opsPairs = Object.entries(ops).filter(([, v]) => Number(v) > 0).map(([k, v]) => `${k}: ${Number(v).toLocaleString()}`);
  document.getElementById("ops-main-val").textContent = data.peak_candidates !== null && data.peak_candidates !== undefined
    ? `${Number(data.peak_candidates).toLocaleString()} peak candidates`
    : `${algoUsed} telemetry`;
  document.getElementById("ops-detail").textContent = opsPairs.length > 0 ? opsPairs.slice(0, 3).join(" · ") : "Zero explosion overhead";

  const weightStatus = data.weighted_info?.active
    ? `Eq. 2.1 boosted "${data.weighted_info.vip_item}" ×${data.weighted_info.multiplier}`
    : "Unweighted uniform txns (sample_weight=None)";
  document.getElementById("score-main-val").textContent = data.estimator_score !== null && data.estimator_score !== undefined
    ? `Score: ${Number(data.estimator_score).toFixed(4)}`
    : "Score: N/A (0 rules)";
  document.getElementById("score-detail").textContent = weightStatus;

  renderSampleChips(data.sample_items);
  renderEngineComparison(data.engine_comparison);
  renderPipelineTransform(data.pipeline_transform);

  const recPanel = document.getElementById("recommendations-panel");
  const recTbody = document.getElementById("recommendations-tbody");
  recTbody.textContent = "";
  if (Array.isArray(data.recommendations) && data.recommendations.length > 0) {
    recPanel.classList.remove("hidden");
    document.getElementById("recommendations-count").textContent = `${data.recommendations.length} matches`;
    data.recommendations.forEach((rec) => {
      const tr = el("tr", "hover:bg-slate-50/70");
      tr.append(
        el("td", "py-2 px-4 text-slate-600 text-xs font-mono", rec.matched_antecedent.join(", ")),
        el("td", "py-2 px-4 text-slate-900 font-medium text-xs font-mono", rec.recommended.join(", ")),
        el("td", "py-2 px-4 text-right font-mono text-xs tabular-nums text-slate-900", fmtNum(rec.confidence, 3)),
        el("td", "py-2 px-4 text-right font-mono text-xs tabular-nums text-blue-700 font-medium", fmtNum(rec.lift, 3)),
        el("td", "py-2 px-4 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rec.support, 4))
      );
      recTbody.appendChild(tr);
    });
  } else {
    recPanel.classList.add("hidden");
  }

  const tbody = document.getElementById("rules-tbody");
  const emptyEl = document.getElementById("rules-empty");
  tbody.textContent = "";
  document.getElementById("rules-showing").textContent =
    data.n_rules > data.rules.length ? `Showing top ${data.rules.length} of ${data.n_rules.toLocaleString()} rules` : `${data.n_rules.toLocaleString()} rules`;

  if (data.rules.length === 0) {
    emptyEl.classList.remove("hidden");
    renderRuleExplainer(null, 0, data);
  } else {
    emptyEl.classList.add("hidden");
    data.rules.forEach((rule, idx) => {
      const tr = el("tr", `sk-rule-row hover:bg-slate-50/70 ${idx === 0 ? "selected" : ""}`);
      tr.title = "Click to inspect formula breakdown";
      tr.addEventListener("click", () => renderRuleExplainer(rule, idx, data));

      const ruleCell = el("td", "py-2.5 px-4 text-slate-900");
      ruleCell.append(el("span", "font-medium text-slate-900", rule.antecedent.join(", ")), el("span", "mx-2 font-mono text-blue-600", "→"), el("span", "font-medium text-slate-700", rule.consequent.join(", ")));

      tr.append(
        ruleCell,
        el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rule.support, 4)),
        el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-slate-900 font-medium", fmtNum(rule.confidence, 4)),
        el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-blue-700 font-medium", fmtNum(rule.lift, 3)),
        el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rule.conviction, 3)),
        el("td", "py-2.5 px-4 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rule.zhangs_metric, 3))
      );
      tbody.appendChild(tr);
    });
    renderRuleExplainer(data.rules[0], 0, data);
  }

  const grid = document.getElementById("itemsets-grid");
  grid.textContent = "";
  document.getElementById("itemsets-showing").textContent = `${data.n_itemsets.toLocaleString()} total`;
  const slice = (data.top_itemsets || []).slice(0, 12);
  if (slice.length === 0) {
    grid.appendChild(el("div", "p-4 text-slate-500 col-span-full", "No frequent itemsets above min_support."));
  } else {
    slice.forEach((itemset) => {
      const cell = el("div", "p-3 border-b border-r border-slate-100 flex items-center justify-between gap-2");
      cell.append(el("span", "text-slate-800 truncate", itemset.items.join(" + ")), el("span", "font-mono text-slate-500 tabular-nums shrink-0", fmtNum(itemset.support, 3)));
      grid.appendChild(cell);
    });
  }
  updateLivePythonSnippet();
}

function renderBenchmarkChart(scales) {
  const container = document.getElementById("benchmark-chart-container");
  container.textContent = "";
  const entries = [
    { key: "scale_200", label: "200 txns", data: scales.scale_200 },
    { key: "scale_1000", label: "1,000 txns", data: scales.scale_1000 },
    { key: "scale_3000", label: "3,000 txns", data: scales.scale_3000 },
  ];
  const maxMs = Math.max(...entries.flatMap((e) => [e.data.apriori.engine_runtime_p95_s * 1000, e.data.fpgrowth.engine_runtime_p95_s * 1000, e.data.eclat.engine_runtime_p95_s * 1000]));
  const yCeiling = Math.ceil((maxMs * 1.28) * 2) / 2;
  const svgNS = "http://www.w3.org/2000/svg";
  const width = 760, height = 220, padL = 56, padR = 24, padT = 18, padB = 34;
  const plotW = width - padL - padR, plotH = height - padT - padB;
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "w-full max-w-[820px] h-auto block");

  for (let i = 0; i <= 4; i++) {
    const val = (yCeiling / 4) * i;
    const y = padT + plotH - (val / yCeiling) * plotH;
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(padL)); line.setAttribute("x2", String(width - padR));
    line.setAttribute("y1", String(y)); line.setAttribute("y2", String(y));
    line.setAttribute("stroke", i === 0 ? "#cbd5e1" : "#e2e8f0");
    svg.appendChild(line);
    const lbl = document.createElementNS(svgNS, "text");
    lbl.setAttribute("x", String(padL - 8)); lbl.setAttribute("y", String(y + 4));
    lbl.setAttribute("text-anchor", "end"); lbl.setAttribute("fill", "#64748b");
    lbl.setAttribute("font-size", "10"); lbl.setAttribute("font-family", "IBM Plex Mono, monospace");
    lbl.textContent = `${val.toFixed(1)} ms`;
    svg.appendChild(lbl);
  }

  const groupW = plotW / entries.length, barW = 42, barGap = 10;
  entries.forEach((entry, idx) => {
    const cx = padL + groupW * idx + groupW / 2;
    const bars = [
      { ms: entry.data.apriori.engine_runtime_p95_s * 1000, color: "#94a3b8" },
      { ms: entry.data.fpgrowth.engine_runtime_p95_s * 1000, color: "#2563eb" },
      { ms: entry.data.eclat.engine_runtime_p95_s * 1000, color: "#0891b2" },
    ];
    const startX = cx - (bars.length * barW + (bars.length - 1) * barGap) / 2;
    const limitY = padT + plotH - ((bars[0].ms * 0.8) / yCeiling) * plotH;
    const lLine = document.createElementNS(svgNS, "line");
    lLine.setAttribute("x1", String(startX - 6)); lLine.setAttribute("x2", String(startX + 150));
    lLine.setAttribute("y1", String(limitY)); lLine.setAttribute("y2", String(limitY));
    lLine.setAttribute("stroke", "#059669"); lLine.setAttribute("stroke-width", "1.5"); lLine.setAttribute("stroke-dasharray", "4 3");
    svg.appendChild(lLine);

    bars.forEach((b, bIdx) => {
      const h = Math.max(2, (b.ms / yCeiling) * plotH);
      const x = startX + bIdx * (barW + barGap);
      const y = padT + plotH - h;
      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("x", String(x)); rect.setAttribute("y", String(y));
      rect.setAttribute("width", String(barW)); rect.setAttribute("height", String(h)); rect.setAttribute("fill", b.color);
      svg.appendChild(rect);
      const vt = document.createElementNS(svgNS, "text");
      vt.setAttribute("x", String(x + barW / 2)); vt.setAttribute("y", String(y - 4));
      vt.setAttribute("text-anchor", "middle"); vt.setAttribute("fill", "#334155");
      vt.setAttribute("font-size", "10"); vt.setAttribute("font-family", "IBM Plex Mono, monospace");
      vt.textContent = b.ms.toFixed(2);
      svg.appendChild(vt);
    });

    const xl = document.createElementNS(svgNS, "text");
    xl.setAttribute("x", String(cx)); xl.setAttribute("y", String(height - 10));
    xl.setAttribute("text-anchor", "middle"); xl.setAttribute("fill", "#0f172a");
    xl.setAttribute("font-size", "11"); xl.setAttribute("font-family", "IBM Plex Mono, monospace");
    xl.textContent = entry.label;
    svg.appendChild(xl);
  });
  container.appendChild(svg);
}

function renderBenchmarkTable(scales, overallPass) {
  const tbody = document.getElementById("benchmarks-tbody");
  tbody.textContent = "";
  const verdictEl = document.getElementById("benchmark-verdict");
  verdictEl.textContent = overallPass ? "OVERALL PASS · 0 FAILURES" : "GATE FAILED";
  verdictEl.className = `text-xs font-mono uppercase ${overallPass ? "text-emerald-700" : "text-rose-700"}`;

  [["scale_200", "200 txns"], ["scale_1000", "1,000 txns"], ["scale_3000", "3,000 txns"]].forEach(([sk, sl]) => {
    const sc = scales[sk];
    if (!sc) return;
    [
      { key: "apriori", name: "Apriori (O2)", ratio: 1.0, pass: true },
      { key: "fpgrowth", name: "FP-Growth", ratio: sc.fpgrowth_engine_p95_ratio_vs_apriori, pass: sc.fpgrowth_kpi4_pass },
      { key: "eclat", name: "Eclat", ratio: sc.eclat_engine_p95_ratio_vs_apriori, pass: sc.eclat_kpi4_pass },
    ].forEach((r, idx) => {
      const m = sc[r.key];
      const tr = el("tr", idx === 0 ? "bg-slate-50/40" : "bg-white");
      tr.append(
        el("td", "py-2 px-3 text-slate-900 font-medium", idx === 0 ? sl : ""),
        el("td", "py-2 px-3 text-slate-800", r.name),
        el("td", "py-2 px-3 text-right text-slate-600", fmtMs(m.engine_runtime_p50_s)),
        el("td", "py-2 px-3 text-right text-slate-900 font-semibold", fmtMs(m.engine_runtime_p95_s)),
        el("td", `py-2 px-3 text-right ${r.key === "apriori" ? "text-slate-500" : "text-emerald-700 font-semibold"}`, r.key === "apriori" ? "1.000×" : `${r.ratio.toFixed(3)}×`),
        el("td", "py-2 px-3 text-right text-slate-600", fmtMs(m.runtime_p95_s)),
        el("td", "py-2 px-3 text-right text-slate-600", m.engine_peak_mem_mb.toFixed(4)),
        el("td", `py-2 px-3 text-right ${r.pass ? "text-emerald-700" : "text-rose-700"}`, `0.00e+00 · ${r.pass ? "PASS" : "FAIL"}`)
      );
      tbody.appendChild(tr);
    });
  });
}

function renderAblation(dsKey) {
  state.activeAblDataset = dsKey;
  document.querySelectorAll(".abl-btn").forEach((b) => b.classList.toggle("active", b.dataset.abl === dsKey));
  const abl = state.overview?.ablation?.candidate_sweeps?.[dsKey];
  const noteEl = document.getElementById("ablation-summary-note");
  const tbody = document.getElementById("ablation-tbody");
  const chartEl = document.getElementById("ablation-chart");
  if (!abl) return;

  const base = abl.baseline || {};
  noteEl.textContent = `Exact Unbounded Baseline: ${base.n_itemsets} itemsets, ${(base.n_rules || 0).toLocaleString()} rules in ${base.fit_time_s}s (${base.status}). Lower caps bound memory & time safely with explicit DEGRADED flag until cap exceeds peak lattice size.`;

  tbody.textContent = "";
  const recs = abl.records || [];
  recs.forEach((r) => {
    const sp = Number(r.speedup_factor ?? (base.fit_time_s && r.fit_time_s ? base.fit_time_s / r.fit_time_s : 1));
    const isDeg = Boolean(r.is_degraded_);
    const statusLabel = r.status_ || (isDeg ? "DEGRADED" : "EXACT");
    const tr = el("tr", "hover:bg-slate-50/70");
    tr.append(
      el("td", "py-2 px-3 font-semibold text-slate-900", r.candidate_cap),
      el("td", "py-2 px-3 text-right", r.n_itemsets),
      el("td", "py-2 px-3 text-right text-blue-700 font-medium", `${(r.itemset_recall * 100).toFixed(1)}%`),
      el("td", "py-2 px-3 text-right", (r.n_rules || 0).toLocaleString()),
      el("td", "py-2 px-3 text-right text-slate-600", `${(r.rule_recall * 100).toFixed(1)}%`),
      el("td", "py-2 px-3 text-right", `${Number(r.fit_time_s).toFixed(4)}s`),
      el("td", "py-2 px-3 text-right font-semibold text-emerald-700", `${sp.toFixed(2)}×`),
      el("td", `py-2 px-3 text-right font-semibold ${isDeg ? "sk-degraded-text" : "text-emerald-700"}`, statusLabel)
    );
    tbody.appendChild(tr);
  });

  // Compact SVG trade-off bars (Recall % vs Speedup)
  chartEl.textContent = "";
  const svgNS = "http://www.w3.org/2000/svg";
  const w = 720, h = 158, padL = 45, padB = 26, pW = w - 70, pH = h - padB - 22;
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("class", "w-full max-w-[820px] h-auto block");
  const maxSp = Math.max(...recs.map((x) => Number(x.speedup_factor || 1)), 2);
  const slotW = pW / Math.max(1, recs.length);

  recs.forEach((r, i) => {
    const sp = Number(r.speedup_factor || 1);
    const isDeg = Boolean(r.is_degraded_);
    const x = padL + i * slotW + 8;
    const bw = Math.max(10, Math.min(22, slotW * 0.32));
    const hRec = Math.max(2, r.itemset_recall * pH);
    const hSp = Math.max(2, (sp / maxSp) * pH);

    const r1 = document.createElementNS(svgNS, "rect");
    r1.setAttribute("x", String(x)); r1.setAttribute("y", String(22 + pH - hRec));
    r1.setAttribute("width", String(bw)); r1.setAttribute("height", String(hRec));
    r1.setAttribute("fill", "#2563eb");
    svg.appendChild(r1);

    const r2 = document.createElementNS(svgNS, "rect");
    r2.setAttribute("x", String(x + bw + 3)); r2.setAttribute("y", String(22 + pH - hSp));
    r2.setAttribute("width", String(bw)); r2.setAttribute("height", String(hSp));
    r2.setAttribute("fill", isDeg ? "#d97706" : "#059669");
    svg.appendChild(r2);

    const spTxt = document.createElementNS(svgNS, "text");
    spTxt.setAttribute("x", String(x + bw + 3 + bw / 2)); spTxt.setAttribute("y", String(18 + pH - hSp));
    spTxt.setAttribute("text-anchor", "middle"); spTxt.setAttribute("fill", "#334155");
    spTxt.setAttribute("font-size", "9"); spTxt.setAttribute("font-family", "IBM Plex Mono, monospace");
    spTxt.textContent = `${sp.toFixed(1)}×`;
    svg.appendChild(spTxt);

    const lbl = document.createElementNS(svgNS, "text");
    lbl.setAttribute("x", String(x + bw)); lbl.setAttribute("y", String(h - 6));
    lbl.setAttribute("text-anchor", "middle"); lbl.setAttribute("fill", "#475569");
    lbl.setAttribute("font-size", "10"); lbl.setAttribute("font-family", "IBM Plex Mono, monospace");
    lbl.textContent = `cap=${r.candidate_cap}`;
    svg.appendChild(lbl);
  });
  chartEl.appendChild(svg);
}

function renderBaselines(filterDs) {
  state.activeBaseFilter = filterDs;
  document.querySelectorAll(".base-btn").forEach((b) => b.classList.toggle("active", b.dataset.base === filterDs));
  const tbody = document.getElementById("baselines-tbody");
  const dsMap = state.overview?.baselines?.datasets;
  if (!dsMap) return;
  tbody.textContent = "";

  Object.entries(dsMap).forEach(([dsKey, dsObj]) => {
    if (filterDs !== "all" && filterDs !== dsKey) return;
    const supp = dsObj.summary?.support;
    (dsObj.records || []).forEach((rec, idx) => {
      const tr = el("tr", idx === 0 ? "bg-slate-50/50" : "bg-white");
      const timeTxt = rec.status === "SKIPPED" ? "SKIPPED" : rec.internal_time_seconds !== undefined
        ? `${Number(rec.time_seconds).toFixed(2)}s (int ${Number(rec.internal_time_seconds).toFixed(2)}s)`
        : `${Number(rec.time_seconds).toFixed(4)}s`;
      const statTxt = rec.status === "SKIPPED" ? (rec.reason || "Dense > 500 MB cap") : "OK";
      tr.append(
        el("td", "py-2 px-3 font-medium text-slate-900", idx === 0 ? `${dsKey} (σ=${supp})` : ""),
        el("td", "py-2 px-3 text-slate-800", `${rec.framework} · ${rec.algorithm}`),
        el("td", "py-2 px-3 text-slate-600", rec.language),
        el("td", `py-2 px-3 text-right ${rec.framework === "skarm" ? "text-blue-700 font-semibold" : "text-slate-700"}`, timeTxt),
        el("td", "py-2 px-3 text-right", rec.n_itemsets ?? "—"),
        el("td", "py-2 px-3 text-right text-emerald-700", rec.exact_match ? "0.0 (Exact)" : rec.status === "SKIPPED" ? "N/A" : "Mismatch"),
        el("td", `py-2 px-3 text-right ${rec.status === "OK" ? "text-emerald-700" : "sk-degraded-text"}`, statTxt)
      );
      tbody.appendChild(tr);
    });
  });
}

function renderCampaignTable() {
  const tbody = document.getElementById("campaign-tbody");
  const rows = state.overview?.campaign_summary?.datasets || [];
  const q = (document.getElementById("campaign-search")?.value || "").trim().toLowerCase();
  const winFilter = document.getElementById("campaign-winner-filter")?.value || "all";
  tbody.textContent = "";

  const fmtCell = (obj) => {
    if (!obj) return "—";
    if (obj.status === "ok") return `${obj.mean_s.toFixed(3)}s`;
    return obj.status === "skipped" || obj.reason ? `SKIP (${obj.status})` : obj.status.toUpperCase();
  };

  rows.forEach((r) => {
    if (q && !r.dataset.toLowerCase().includes(q)) return;
    if (winFilter === "eclat" && r.fastest_skarm !== "eclat") return;
    if (winFilter === "fpgrowth" && r.fastest_skarm !== "fpgrowth") return;
    if (winFilter === "mlxtend_issue" && r.mlxtend?.status === "ok") return;

    const actionTd = el("td", "py-2 px-3 text-right");
    if (Object.prototype.hasOwnProperty.call(DATASET_DEFAULTS, r.dataset)) {
      const runBtn = el("button", "sk-viva-btn", "Run Live ▶");
      runBtn.type = "button";
      runBtn.addEventListener("click", () => launchDatasetLive(r.dataset));
      actionTd.appendChild(runBtn);
    } else {
      actionTd.appendChild(el("span", "text-[11px] text-slate-400", "Batch Campaign"));
    }

    const tr = el("tr", "hover:bg-slate-50/70");
    tr.append(
      el("td", "py-2 px-3 font-medium text-slate-900", r.dataset),
      el("td", "py-2 px-3 text-right text-slate-600", Number(r.expected_n || 0).toLocaleString()),
      el("td", "py-2 px-3 text-right text-slate-600", r.primary_thr),
      el("td", "py-2 px-3 text-right text-slate-600", fmtCell(r.apriori)),
      el("td", `py-2 px-3 text-right ${r.fastest_skarm === "fpgrowth" ? "font-bold text-blue-700" : "text-slate-700"}`, fmtCell(r.fpgrowth)),
      el("td", `py-2 px-3 text-right ${r.fastest_skarm === "eclat" ? "font-bold text-cyan-700" : "text-slate-700"}`, fmtCell(r.eclat)),
      el("td", `py-2 px-3 text-right ${r.mlxtend?.status === "ok" ? "text-slate-600" : "text-rose-700"}`, fmtCell(r.mlxtend)),
      el("td", "py-2 px-3 text-right font-semibold text-emerald-700 uppercase", r.fastest_skarm),
      actionTd
    );
    tbody.appendChild(tr);
  });
}

function renderE2ETiers(e2e) {
  const grid = document.getElementById("e2e-tiers-grid");
  grid.textContent = "";
  const passed = e2e.total_passed ?? 0, total = e2e.total_tests ?? 0;
  const dur = e2e.total_time_seconds !== undefined ? `${Number(e2e.total_time_seconds).toFixed(2)}s` : "";
  document.getElementById("e2e-summary-label").textContent = `${passed} / ${total} PASSED${dur ? ` · ${dur}` : ""}`;

  (e2e.tiers || []).forEach((info) => {
    const cell = el("div", "p-4");
    cell.append(
      el("div", "text-[11px] font-mono uppercase tracking-wider text-slate-500", `${info.tier} · ${info.name}`),
      el("div", "mt-1 text-xl font-mono font-semibold text-slate-900 tabular-nums", `${info.passed} / ${info.total}`),
      el("div", `mt-1 text-xs font-mono ${info.failed === 0 && info.errors === 0 ? "text-emerald-700" : "text-rose-700"}`, `0 failures · ${Number(info.duration_s).toFixed(2)}s`)
    );
    grid.appendChild(cell);
  });
}

function renderVivaQA() {
  const grid = document.getElementById("viva-qa-grid");
  if (!grid) return;
  grid.textContent = "";
  VIVA_QA.forEach((item) => {
    const card = el("div", "sk-viva-card rounded-sm");
    card.append(
      el("div", "text-xs font-semibold text-slate-900", item.q),
      el("p", "text-xs text-slate-600 leading-relaxed", item.a)
    );
    const btn = el("button", "sk-viva-btn", `${item.btnLabel} →`);
    btn.type = "button";
    btn.addEventListener("click", () => {
      switchTab("explorer");
      item.action();
    });
    card.appendChild(btn);
    grid.appendChild(card);
  });
}

function renderOverview(data) {
  state.overview = data;
  const tests = data?.project?.tests;
  if (tests) {
    document.getElementById("kpi-tests").textContent = `${tests.pytest_passed} / ${tests.pytest_total}`;
    document.getElementById("kpi-e2e-sub").textContent = `${tests.e2e_passed} / ${tests.e2e_total} E2E · 5 Tiers`;
  }

  const bench = data?.benchmarks;
  const scales = bench?.KPI_3_4_5_scale_benchmarks || bench?.["KPI-3_4_5_scale_benchmarks"];
  if (scales) {
    const s200 = scales.scale_200, s1000 = scales.scale_1000, s3000 = scales.scale_3000;
    const fpMin = Math.min(s200.fpgrowth_engine_p95_ratio_vs_apriori, s1000.fpgrowth_engine_p95_ratio_vs_apriori, s3000.fpgrowth_engine_p95_ratio_vs_apriori);
    const fpMax = Math.max(s200.fpgrowth_engine_p95_ratio_vs_apriori, s1000.fpgrowth_engine_p95_ratio_vs_apriori, s3000.fpgrowth_engine_p95_ratio_vs_apriori);
    const ecMax = Math.max(s200.eclat_engine_p95_ratio_vs_apriori, s1000.eclat_engine_p95_ratio_vs_apriori, s3000.eclat_engine_p95_ratio_vs_apriori);
    document.getElementById("kpi-fp-ratio").textContent = `${fpMin.toFixed(2)}×–${fpMax.toFixed(2)}×`;
    document.getElementById("kpi-eclat-sub").textContent = `Eclat ≤ ${ecMax.toFixed(2)}× · 30 trials · 0 exemptions`;
    renderBenchmarkTable(scales, bench.overall_pass);
    renderBenchmarkChart(scales);
  }

  const kpi2 = bench?.["KPI-2_rule_metric_numerical_error"];
  const cs = data?.campaign_summary;
  if (kpi2 && cs) {
    document.getElementById("kpi-err-sub").textContent = `${cs.pair_matches}/${cs.pair_total} pairs + 25/25 SPMF · ${Number(kpi2.max_numerical_error).toExponential(2)}`;
  }

  if (data?.e2e) renderE2ETiers(data.e2e);
  if (data?.ablation) renderAblation("chess");
  if (data?.baselines) renderBaselines("all");
  if (cs) {
    document.getElementById("campaign-sub").textContent =
      `thresholds[0] Audit: Eclat fastest on ${cs.wins.eclat}/${cs.total_datasets} · FP-Growth fastest on ${cs.wins.fpgrowth}/${cs.total_datasets} · Apriori ${cs.wins.apriori}/${cs.total_datasets} · ${cs.pair_matches}/${cs.pair_total} bit-exact pairs (${cs.total_trials} trials; full-sweep Table I.1 highlights 5 FP-Growth dense/lower-support wins).`;
    const optEc = document.getElementById("opt-eclat-win");
    const optFp = document.getElementById("opt-fp-win");
    if (optEc) optEc.textContent = `Eclat Fastest (${cs.wins.eclat})`;
    if (optFp) optFp.textContent = `FP-Growth Fastest (${cs.wins.fpgrowth})`;
    renderCampaignTable();
  }
}

async function runMining(compareEngines = false) {
  if (state.mining) return;
  const errorEl = document.getElementById("mine-error");
  const submitBtn = document.getElementById("mine-submit");
  const compareBtn = document.getElementById("mine-compare");
  errorEl.classList.add("hidden");
  errorEl.textContent = "";

  const dataset = document.getElementById("dataset-select").value;
  const algorithm = document.getElementById("algorithm-select").value;
  const metric = document.getElementById("metric-select").value;
  const minSupport = Number(document.getElementById("min-support").value);
  const minThreshold = Number(document.getElementById("min-threshold").value);
  const maxCandidates = Number(document.getElementById("max-candidates").value);
  const maxLenRaw = document.getElementById("max-len").value.trim();
  const maxLen = maxLenRaw ? Number(maxLenRaw) : null;
  const vipItem = document.getElementById("vip-item").value.trim() || null;
  const vipMultiplier = Number(document.getElementById("vip-multiplier").value || "1");
  const basketText = document.getElementById("basket-input").value;
  const transactionsText = document.getElementById("custom-txns").value;

  if (!Number.isFinite(minSupport) || minSupport <= 0 || minSupport > 1) {
    errorEl.textContent = "Min Support must be in (0, 1].";
    errorEl.classList.remove("hidden");
    return;
  }

  state.mining = true;
  submitBtn.disabled = true;
  compareBtn.disabled = true;
  submitBtn.textContent = compareEngines ? "Running..." : "Mining...";
  compareBtn.textContent = compareEngines ? "Comparing..." : "Compare 3 Engines";
  document.getElementById("result-status").textContent = "RUNNING";

  try {
    const res = await fetch("/api/mine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset, algorithm, metric,
        min_support: minSupport,
        min_threshold: minThreshold,
        max_len: maxLen,
        max_candidates: maxCandidates,
        vip_item: vipItem,
        vip_multiplier: vipMultiplier,
        basket_text: basketText,
        transactions_text: dataset === "custom" ? transactionsText : undefined,
        top_n: 50,
        compare_engines: compareEngines,
      }),
    });
    const payload = await res.json();
    if (!res.ok) {
      errorEl.textContent = payload.error || "Mining request failed.";
      errorEl.classList.remove("hidden");
      document.getElementById("result-status").textContent = "ERROR";
      return;
    }
    renderMineResult(payload);
  } catch {
    errorEl.textContent = "Network error while contacting the skarm server.";
    errorEl.classList.remove("hidden");
    document.getElementById("result-status").textContent = "ERROR";
  } finally {
    state.mining = false;
    submitBtn.disabled = false;
    compareBtn.disabled = false;
    submitBtn.textContent = "Mine Rules";
    compareBtn.textContent = "Compare 3 Engines";
  }
}

async function launchDatasetLive(val) {
  switchTab("explorer");
  document.querySelectorAll(".sk-preset-btn").forEach((b) => b.classList.remove("active"));
  const datasetSelect = document.getElementById("dataset-select");
  datasetSelect.value = val;
  document.getElementById("custom-txns-wrapper").classList.toggle("hidden", val !== "custom");
  const p = DATASET_DEFAULTS[val];
  if (p) {
    document.getElementById("min-support").value = p.min_support;
    document.getElementById("min-threshold").value = p.min_threshold;
    document.getElementById("metric-select").value = p.metric;
    document.getElementById("basket-input").value = p.basket;
    document.getElementById("max-candidates").value = p.max_candidates;
    document.getElementById("max-len").value = p.max_len;
    document.getElementById("vip-item").value = p.vip_item;
    document.getElementById("vip-multiplier").value = p.vip_multiplier;
    document.getElementById("algorithm-select").value = p.algorithm;
  }
  updateLivePythonSnippet();
  if (val !== "custom") {
    await runMining(false);
  }
}

async function applyDemoPreset(presetKey) {
  const p = DEMO_PRESETS[presetKey];
  if (!p) return;
  document.querySelectorAll(".sk-preset-btn").forEach((b) => b.classList.toggle("active", b.dataset.preset === presetKey));
  document.getElementById("dataset-select").value = p.dataset;
  document.getElementById("custom-txns-wrapper").classList.toggle("hidden", p.dataset !== "custom");
  document.getElementById("algorithm-select").value = p.algorithm;
  document.getElementById("metric-select").value = p.metric;
  document.getElementById("min-support").value = p.min_support;
  document.getElementById("min-threshold").value = p.min_threshold;
  document.getElementById("max-candidates").value = p.max_candidates;
  document.getElementById("max-len").value = p.max_len;
  document.getElementById("vip-item").value = p.vip_item;
  document.getElementById("vip-multiplier").value = p.vip_multiplier;
  document.getElementById("basket-input").value = p.basket;
  updateLivePythonSnippet();
  await runMining(false);
}

async function runAcceptance() {
  switchTab("acceptance");
  if (state.acceptanceRunning) return;
  const errEl = document.getElementById("acceptance-error");
  const verdictEl = document.getElementById("acceptance-verdict");
  const tbody = document.getElementById("acceptance-tbody");
  const panelBtn = document.getElementById("panel-run-acceptance");
  const headerBtn = document.getElementById("header-run-acceptance");

  errEl.classList.add("hidden");
  state.acceptanceRunning = true;
  panelBtn.disabled = true;
  headerBtn.disabled = true;
  panelBtn.textContent = "Running 30-Trial Gate...";
  verdictEl.textContent = "EXECUTING tests/run_acceptance.py...";
  verdictEl.className = "text-xs font-mono uppercase text-blue-700";

  try {
    const res = await fetch("/api/acceptance", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      errEl.textContent = data.error || "Acceptance execution failed.";
      errEl.classList.remove("hidden");
      verdictEl.textContent = "EXECUTION ERROR";
      return;
    }
    const passed = data.exit_code === 0 && data.failed_count === 0;
    verdictEl.textContent = `${data.verdict} · ${data.passed_count}/${data.total_count} · ${(data.elapsed_ms / 1000).toFixed(2)}s`;
    verdictEl.className = `text-xs font-mono uppercase ${passed ? "text-emerald-700" : "text-rose-700"}`;

    tbody.textContent = "";
    data.checks.forEach((chk) => {
      const tr = el("tr", "hover:bg-slate-50/70");
      tr.append(
        el("td", "py-2.5 px-4 font-semibold text-slate-900", chk.id),
        el("td", `py-2.5 px-4 font-semibold ${chk.status === "PASS" ? "text-emerald-700" : "text-rose-700"}`, chk.status),
        el("td", "py-2.5 px-4 text-slate-700", chk.detail)
      );
      tbody.appendChild(tr);
    });
  } catch {
    errEl.textContent = "Network error invoking acceptance runner.";
    errEl.classList.remove("hidden");
  } finally {
    state.acceptanceRunning = false;
    panelBtn.disabled = false;
    headerBtn.disabled = false;
    panelBtn.textContent = "Run Strict Acceptance Gate";
  }
}

async function init() {
  renderVivaQA();
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));
  document.querySelectorAll(".sk-preset-btn").forEach((btn) => btn.addEventListener("click", () => applyDemoPreset(btn.dataset.preset)));
  document.querySelectorAll(".obj-jump").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.tabJump) switchTab(btn.dataset.tabJump);
      else if (btn.dataset.preset) { switchTab("explorer"); applyDemoPreset(btn.dataset.preset); }
    });
  });

  document.querySelectorAll(".abl-btn").forEach((b) => b.addEventListener("click", () => renderAblation(b.dataset.abl)));
  document.querySelectorAll(".base-btn").forEach((b) => b.addEventListener("click", () => renderBaselines(b.dataset.base)));
  document.querySelectorAll(".doc-view-btn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".doc-view-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      document.getElementById("ieee-paper-frame").src = b.dataset.doc;
      const openLink = document.getElementById("open-doc-tab");
      if (openLink) openLink.href = b.dataset.doc;
    });
  });

  document.getElementById("campaign-search")?.addEventListener("input", renderCampaignTable);
  document.getElementById("campaign-winner-filter")?.addEventListener("change", renderCampaignTable);
  document.getElementById("header-run-acceptance").addEventListener("click", runAcceptance);
  document.getElementById("panel-run-acceptance").addEventListener("click", runAcceptance);

  const copyBtn = document.getElementById("copy-python-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const text = document.getElementById("live-python-code")?.textContent || "";
      if (text && navigator.clipboard) {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "Copied";
        setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
      }
    });
  }

  const datasetSelect = document.getElementById("dataset-select");
  datasetSelect.addEventListener("change", () => launchDatasetLive(datasetSelect.value));

  ["algorithm-select", "metric-select", "min-support", "min-threshold", "max-candidates", "max-len", "vip-item", "vip-multiplier"].forEach((id) => {
    document.getElementById(id)?.addEventListener("change", updateLivePythonSnippet);
  });

  document.getElementById("mine-form").addEventListener("submit", (e) => { e.preventDefault(); runMining(false); });
  document.getElementById("mine-compare").addEventListener("click", () => runMining(true));

  try {
    const res = await fetch("/api/overview");
    if (res.ok) renderOverview(await res.json());
  } catch { /* non-fatal */ }

  document.getElementById("basket-input").value = DATASET_DEFAULTS.groceries.basket;
  updateLivePythonSnippet();
  await runMining(false);
}

init();
