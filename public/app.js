const DATASET_DEFAULTS = {
  groceries: { min_support: "0.02", min_threshold: "0.25", metric: "confidence", basket: "yogurt, rolls/buns" },
  demo: { min_support: "0.2", min_threshold: "0.5", metric: "confidence", basket: "milk, bread" },
  market_basket: { min_support: "0.015", min_threshold: "0.25", metric: "confidence", basket: "mineral water" },
  tic_tac_toe: { min_support: "0.25", min_threshold: "0.7", metric: "confidence", basket: "" },
  mushroom: { min_support: "0.55", min_threshold: "0.85", metric: "confidence", basket: "" },
  custom: { min_support: "0.3", min_threshold: "0.5", metric: "confidence", basket: "milk" },
};

const state = {
  activeTab: "explorer",
  overview: null,
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
    if (panel) {
      panel.classList.toggle("hidden", name !== tabName);
    }
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
    const s200 = scales.scale_200;
    const s1000 = scales.scale_1000;
    const s3000 = scales.scale_3000;
    const fpMin = Math.min(
      s200.fpgrowth_engine_p95_ratio_vs_apriori,
      s1000.fpgrowth_engine_p95_ratio_vs_apriori,
      s3000.fpgrowth_engine_p95_ratio_vs_apriori
    );
    const fpMax = Math.max(
      s200.fpgrowth_engine_p95_ratio_vs_apriori,
      s1000.fpgrowth_engine_p95_ratio_vs_apriori,
      s3000.fpgrowth_engine_p95_ratio_vs_apriori
    );
    const ecMax = Math.max(
      s200.eclat_engine_p95_ratio_vs_apriori,
      s1000.eclat_engine_p95_ratio_vs_apriori,
      s3000.eclat_engine_p95_ratio_vs_apriori
    );
    document.getElementById("kpi-fp-ratio").textContent = `${fpMin.toFixed(2)}×–${fpMax.toFixed(2)}×`;
    document.getElementById("kpi-eclat-sub").textContent = `Eclat ≤ ${ecMax.toFixed(2)}× · 30 trials · 0 exemptions`;

    renderBenchmarkTable(scales, bench.overall_pass);
    renderBenchmarkChart(scales);
  }

  const kpi2 = bench?.["KPI-2_rule_metric_numerical_error"];
  if (kpi2) {
    document.getElementById("kpi-err-sub").textContent = `Rule error ${Number(kpi2.max_numerical_error).toExponential(2)} · 100% Oracle`;
  }

  const kpi6 = bench?.["KPI-6_estimator_api_compliance"];
  if (kpi6) {
    document.getElementById("kpi-api").textContent = `${kpi6.checks_passed} / ${kpi6.checks_total}`;
  }

  if (data?.e2e) {
    renderE2ETiers(data.e2e);
  }
}

function renderBenchmarkChart(scales) {
  const container = document.getElementById("benchmark-chart-container");
  container.textContent = "";

  const entries = [
    { key: "scale_200", label: "200 txns", data: scales.scale_200 },
    { key: "scale_1000", label: "1,000 txns", data: scales.scale_1000 },
    { key: "scale_3000", label: "3,000 txns", data: scales.scale_3000 },
  ];

  const maxMs = Math.max(
    ...entries.flatMap((e) => [
      e.data.apriori.engine_runtime_p95_s * 1000,
      e.data.fpgrowth.engine_runtime_p95_s * 1000,
      e.data.eclat.engine_runtime_p95_s * 1000,
    ])
  );
  const yCeiling = Math.ceil((maxMs * 1.28) * 2) / 2;

  const svgNS = "http://www.w3.org/2000/svg";
  const width = 760;
  const height = 240;
  const padL = 56;
  const padR = 24;
  const padT = 20;
  const padB = 38;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "w-full max-w-[820px] h-auto block");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Grouped bar chart of Engine p95 runtimes in milliseconds across 200, 1000, and 3000 transaction scales.");

  // Horizontal grid lines
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const val = (yCeiling / ticks) * i;
    const y = padT + plotH - (val / yCeiling) * plotH;
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(padL));
    line.setAttribute("x2", String(width - padR));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", i === 0 ? "#cbd5e1" : "#e2e8f0");
    line.setAttribute("stroke-width", "1");
    svg.appendChild(line);

    const lbl = document.createElementNS(svgNS, "text");
    lbl.setAttribute("x", String(padL - 10));
    lbl.setAttribute("y", String(y + 4));
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("fill", "#64748b");
    lbl.setAttribute("font-size", "11");
    lbl.setAttribute("font-family", "IBM Plex Mono, monospace");
    lbl.textContent = `${val.toFixed(1)} ms`;
    svg.appendChild(lbl);
  }

  const groupW = plotW / entries.length;
  const barW = 42;
  const barGap = 10;

  entries.forEach((entry, idx) => {
    const cx = padL + groupW * idx + groupW / 2;
    const bars = [
      { name: "Apriori", ms: entry.data.apriori.engine_runtime_p95_s * 1000, color: "#94a3b8" },
      { name: "FP-Growth", ms: entry.data.fpgrowth.engine_runtime_p95_s * 1000, color: "#2563eb" },
      { name: "Eclat", ms: entry.data.eclat.engine_runtime_p95_s * 1000, color: "#0891b2" },
    ];
    const totalBarsW = bars.length * barW + (bars.length - 1) * barGap;
    const startX = cx - totalBarsW / 2;

    // 0.80x Apriori KPI-4 target line per scale group
    const limitMs = bars[0].ms * 0.8;
    const limitY = padT + plotH - (limitMs / yCeiling) * plotH;
    const limitLine = document.createElementNS(svgNS, "line");
    limitLine.setAttribute("x1", String(startX - 6));
    limitLine.setAttribute("x2", String(startX + totalBarsW + 6));
    limitLine.setAttribute("y1", String(limitY));
    limitLine.setAttribute("y2", String(limitY));
    limitLine.setAttribute("stroke", "#059669");
    limitLine.setAttribute("stroke-width", "1.5");
    limitLine.setAttribute("stroke-dasharray", "4 3");
    svg.appendChild(limitLine);

    const limitLbl = document.createElementNS(svgNS, "text");
    limitLbl.setAttribute("x", String(startX + totalBarsW + 8));
    limitLbl.setAttribute("y", String(limitY + 3));
    limitLbl.setAttribute("fill", "#059669");
    limitLbl.setAttribute("font-size", "10");
    limitLbl.setAttribute("font-family", "IBM Plex Mono, monospace");
    limitLbl.textContent = "0.80×";
    svg.appendChild(limitLbl);

    bars.forEach((bar, bIdx) => {
      const h = Math.max(2, (bar.ms / yCeiling) * plotH);
      const x = startX + bIdx * (barW + barGap);
      const y = padT + plotH - h;

      const rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("x", String(x));
      rect.setAttribute("y", String(y));
      rect.setAttribute("width", String(barW));
      rect.setAttribute("height", String(h));
      rect.setAttribute("fill", bar.color);
      svg.appendChild(rect);

      const valText = document.createElementNS(svgNS, "text");
      valText.setAttribute("x", String(x + barW / 2));
      valText.setAttribute("y", String(y - 5));
      valText.setAttribute("text-anchor", "middle");
      valText.setAttribute("fill", "#334155");
      valText.setAttribute("font-size", "10");
      valText.setAttribute("font-family", "IBM Plex Mono, monospace");
      valText.textContent = bar.ms.toFixed(2);
      svg.appendChild(valText);
    });

    const xLbl = document.createElementNS(svgNS, "text");
    xLbl.setAttribute("x", String(cx));
    xLbl.setAttribute("y", String(height - 12));
    xLbl.setAttribute("text-anchor", "middle");
    xLbl.setAttribute("fill", "#0f172a");
    xLbl.setAttribute("font-size", "12");
    xLbl.setAttribute("font-family", "IBM Plex Mono, monospace");
    xLbl.textContent = entry.label;
    svg.appendChild(xLbl);
  });

  container.appendChild(svg);
}

function renderBenchmarkTable(scales, overallPass) {
  const tbody = document.getElementById("benchmarks-tbody");
  tbody.textContent = "";
  const verdictEl = document.getElementById("benchmark-verdict");
  verdictEl.textContent = overallPass ? "OVERALL PASS · 0 FAILURES" : "GATE FAILED";
  verdictEl.className = overallPass
    ? "text-xs font-mono uppercase text-emerald-700"
    : "text-xs font-mono uppercase text-rose-700";

  const order = [
    ["scale_200", "200 txns"],
    ["scale_1000", "1,000 txns"],
    ["scale_3000", "3,000 txns"],
  ];

  for (const [scaleKey, scaleLabel] of order) {
    const sc = scales[scaleKey];
    if (!sc) continue;
    const rows = [
      { key: "apriori", name: "Apriori (O2)", ratio: 1.0, pass: true },
      { key: "fpgrowth", name: "FP-Growth", ratio: sc.fpgrowth_engine_p95_ratio_vs_apriori, pass: sc.fpgrowth_kpi4_pass },
      { key: "eclat", name: "Eclat", ratio: sc.eclat_engine_p95_ratio_vs_apriori, pass: sc.eclat_kpi4_pass },
    ];

    rows.forEach((r, idx) => {
      const m = sc[r.key];
      const tr = el("tr", idx === 0 ? "bg-slate-50/40" : "bg-white");
      tr.appendChild(el("td", "py-2.5 px-4 text-slate-900 font-medium", idx === 0 ? scaleLabel : ""));
      tr.appendChild(el("td", "py-2.5 px-3 text-slate-800", r.name));
      tr.appendChild(el("td", "py-2.5 px-3 text-right text-slate-600", fmtMs(m.engine_runtime_p50_s)));
      tr.appendChild(el("td", "py-2.5 px-3 text-right text-slate-900 font-semibold", fmtMs(m.engine_runtime_p95_s)));
      tr.appendChild(
        el(
          "td",
          `py-2.5 px-3 text-right ${r.key === "apriori" ? "text-slate-500" : r.pass ? "text-emerald-700 font-semibold" : "text-rose-700 font-semibold"}`,
          r.key === "apriori" ? "1.000×" : `${r.ratio.toFixed(3)}×`
        )
      );
      tr.appendChild(el("td", "py-2.5 px-3 text-right text-slate-600", fmtMs(m.runtime_p95_s)));
      tr.appendChild(el("td", "py-2.5 px-3 text-right text-slate-600", m.engine_peak_mem_mb.toFixed(4)));
      tr.appendChild(
        el(
          "td",
          `py-2.5 px-4 text-right ${r.pass ? "text-emerald-700" : "text-rose-700"}`,
          `0.00e+00 · ${r.pass ? "PASS" : "FAIL"}`
        )
      );
      tbody.appendChild(tr);
    });
  }
}

function renderE2ETiers(e2e) {
  const grid = document.getElementById("e2e-tiers-grid");
  grid.textContent = "";
  const summaryEl = document.getElementById("e2e-summary-label");
  const passed = e2e.total_passed ?? 0;
  const total = e2e.total_tests ?? 0;
  const dur = e2e.total_time_seconds !== undefined ? `${Number(e2e.total_time_seconds).toFixed(2)}s` : "";
  summaryEl.textContent = `${passed} / ${total} PASSED${dur ? ` · ${dur}` : ""}`;

  const tiersList = Array.isArray(e2e.tiers) ? e2e.tiers : [];
  tiersList.forEach((info) => {
    const cell = el("div", "p-4");
    cell.appendChild(el("div", "text-[11px] font-mono uppercase tracking-wider text-slate-500", `${info.tier} · ${info.name}`));
    cell.appendChild(el("div", "mt-1 text-xl font-mono font-semibold text-slate-900 tabular-nums", `${info.passed} / ${info.total}`));
    cell.appendChild(
      el(
        "div",
        `mt-1 text-xs font-mono ${info.failed === 0 && info.errors === 0 ? "text-emerald-700" : "text-rose-700"}`,
        info.failed === 0 && info.errors === 0
          ? `0 failures · ${Number(info.duration_s).toFixed(2)}s`
          : `${info.failed} failed · ${info.errors} errors`
      )
    );
    grid.appendChild(cell);
  });
}

function renderMineResult(data) {
  document.getElementById("result-dataset").textContent = data.dataset_label;
  const s = data.summary;
  document.getElementById("result-meta").textContent =
    `${s.n_transactions.toLocaleString()} txns · ${s.n_unique_items.toLocaleString()} items · avg len ${s.avg_transaction_length} · ${data.n_itemsets.toLocaleString()} itemsets · ${data.n_rules.toLocaleString()} rules`;

  const statusEl = document.getElementById("result-status");
  const isExact = data.status === "EXACT";
  statusEl.textContent = `${String(data.algorithm_used).toUpperCase()} · ${data.status} · ${data.elapsed_ms.toFixed(1)} ms`;
  statusEl.className = `text-xs font-mono uppercase tracking-wider tabular-nums ${isExact ? "text-emerald-700" : "text-amber-700"}`;

  // Basket recommendations
  const recPanel = document.getElementById("recommendations-panel");
  const recTbody = document.getElementById("recommendations-tbody");
  recTbody.textContent = "";
  if (Array.isArray(data.recommendations) && data.recommendations.length > 0) {
    recPanel.classList.remove("hidden");
    document.getElementById("recommendations-count").textContent = `${data.recommendations.length} matches`;
    data.recommendations.forEach((rec) => {
      const tr = el("tr", "hover:bg-slate-50/70");
      tr.appendChild(el("td", "py-2 px-4 text-slate-600 text-xs font-mono", rec.matched_antecedent.join(", ")));
      tr.appendChild(el("td", "py-2 px-4 text-slate-900 font-medium text-xs font-mono", rec.recommended.join(", ")));
      tr.appendChild(el("td", "py-2 px-4 text-right font-mono text-xs tabular-nums text-slate-900", fmtNum(rec.confidence, 3)));
      tr.appendChild(el("td", "py-2 px-4 text-right font-mono text-xs tabular-nums text-blue-700 font-medium", fmtNum(rec.lift, 3)));
      tr.appendChild(el("td", "py-2 px-4 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rec.support, 4)));
      recTbody.appendChild(tr);
    });
  } else {
    recPanel.classList.add("hidden");
  }

  // Rules table
  const tbody = document.getElementById("rules-tbody");
  const emptyEl = document.getElementById("rules-empty");
  tbody.textContent = "";
  document.getElementById("rules-showing").textContent =
    data.n_rules > data.rules.length
      ? `Showing top ${data.rules.length} of ${data.n_rules.toLocaleString()} rules`
      : `${data.n_rules.toLocaleString()} rules`;

  if (data.rules.length === 0) {
    emptyEl.classList.remove("hidden");
  } else {
    emptyEl.classList.add("hidden");
    data.rules.forEach((rule) => {
      const tr = el("tr", "hover:bg-slate-50/70");
      const ruleCell = el("td", "py-2.5 px-4 text-slate-900");
      const antSpan = el("span", "font-medium text-slate-900", rule.antecedent.join(", "));
      const arrowSpan = el("span", "mx-2 font-mono text-blue-600", "→");
      const conSpan = el("span", "font-medium text-slate-700", rule.consequent.join(", "));
      ruleCell.append(antSpan, arrowSpan, conSpan);

      tr.appendChild(ruleCell);
      tr.appendChild(el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rule.support, 4)));
      tr.appendChild(el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-slate-900 font-medium", fmtNum(rule.confidence, 4)));
      tr.appendChild(el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-blue-700 font-medium", fmtNum(rule.lift, 3)));
      tr.appendChild(el("td", "py-2.5 px-3 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rule.conviction, 3)));
      tr.appendChild(el("td", "py-2.5 px-4 text-right font-mono text-xs tabular-nums text-slate-600", fmtNum(rule.zhangs_metric, 3)));
      tbody.appendChild(tr);
    });
  }

  // Top frequent itemsets
  const grid = document.getElementById("itemsets-grid");
  grid.textContent = "";
  document.getElementById("itemsets-showing").textContent = `${data.n_itemsets.toLocaleString()} total`;
  const slice = (data.top_itemsets || []).slice(0, 12);
  if (slice.length === 0) {
    grid.appendChild(el("div", "p-4 text-slate-500 col-span-full", "No frequent itemsets above min_support."));
  } else {
    slice.forEach((itemset) => {
      const cell = el("div", "p-3 border-b border-r border-slate-100 flex items-center justify-between gap-2");
      cell.appendChild(el("span", "text-slate-800 truncate", itemset.items.join(" + ")));
      cell.appendChild(el("span", "font-mono text-slate-500 tabular-nums shrink-0", fmtNum(itemset.support, 3)));
      grid.appendChild(cell);
    });
  }
}

async function runMining() {
  if (state.mining) return;
  const errorEl = document.getElementById("mine-error");
  const submitBtn = document.getElementById("mine-submit");
  errorEl.classList.add("hidden");
  errorEl.textContent = "";

  const dataset = document.getElementById("dataset-select").value;
  const algorithm = document.getElementById("algorithm-select").value;
  const metric = document.getElementById("metric-select").value;
  const minSupport = Number(document.getElementById("min-support").value);
  const minThreshold = Number(document.getElementById("min-threshold").value);
  const maxCandidates = Number(document.getElementById("max-candidates").value);
  const basketText = document.getElementById("basket-input").value;
  const transactionsText = document.getElementById("custom-txns").value;

  if (!Number.isFinite(minSupport) || minSupport <= 0 || minSupport > 1) {
    errorEl.textContent = "Min Support must be a number in the interval (0, 1].";
    errorEl.classList.remove("hidden");
    return;
  }
  if (!Number.isFinite(minThreshold)) {
    errorEl.textContent = "Min Threshold must be a valid finite number.";
    errorEl.classList.remove("hidden");
    return;
  }
  if (!Number.isInteger(maxCandidates) || maxCandidates < 1) {
    errorEl.textContent = "Max Candidates Budget must be a positive integer.";
    errorEl.classList.remove("hidden");
    return;
  }

  state.mining = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Mining...";
  document.getElementById("result-status").textContent = "RUNNING";

  try {
    const res = await fetch("/api/mine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        algorithm,
        metric,
        min_support: minSupport,
        min_threshold: minThreshold,
        max_candidates: maxCandidates,
        basket_text: basketText,
        transactions_text: dataset === "custom" ? transactionsText : undefined,
        top_n: 50,
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
    submitBtn.textContent = "Mine Association Rules";
  }
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
  errEl.textContent = "";
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
      errEl.textContent = data.error || "Acceptance gate execution failed.";
      errEl.classList.remove("hidden");
      verdictEl.textContent = "EXECUTION ERROR";
      verdictEl.className = "text-xs font-mono uppercase text-rose-700";
      return;
    }

    const passed = data.exit_code === 0 && data.failed_count === 0;
    verdictEl.textContent = `${data.verdict} · ${data.passed_count}/${data.total_count} · ${(data.elapsed_ms / 1000).toFixed(2)}s`;
    verdictEl.className = `text-xs font-mono uppercase ${passed ? "text-emerald-700" : "text-rose-700"}`;

    tbody.textContent = "";
    data.checks.forEach((chk) => {
      const tr = el("tr", "hover:bg-slate-50/70");
      tr.appendChild(el("td", "py-2.5 px-4 font-semibold text-slate-900", chk.id));
      tr.appendChild(
        el(
          "td",
          `py-2.5 px-4 font-semibold ${chk.status === "PASS" ? "text-emerald-700" : "text-rose-700"}`,
          chk.status
        )
      );
      tr.appendChild(el("td", "py-2.5 px-4 text-slate-700", chk.detail));
      tbody.appendChild(tr);
    });
  } catch {
    errEl.textContent = "Network error while invoking acceptance runner.";
    errEl.classList.remove("hidden");
    verdictEl.textContent = "ERROR";
    verdictEl.className = "text-xs font-mono uppercase text-rose-700";
  } finally {
    state.acceptanceRunning = false;
    panelBtn.disabled = false;
    headerBtn.disabled = false;
    panelBtn.textContent = "Run Strict Acceptance Gate";
  }
}

async function init() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  document.getElementById("header-run-acceptance").addEventListener("click", runAcceptance);
  document.getElementById("panel-run-acceptance").addEventListener("click", runAcceptance);
  const printPaperBtn = document.getElementById("print-paper-btn");
  if (printPaperBtn) {
    printPaperBtn.addEventListener("click", () => {
      const frame = document.getElementById("ieee-paper-frame");
      if (frame && frame.contentWindow) {
        frame.contentWindow.focus();
        frame.contentWindow.print();
      }
    });
  }

  const datasetSelect = document.getElementById("dataset-select");
  const customWrapper = document.getElementById("custom-txns-wrapper");
  datasetSelect.addEventListener("change", () => {
    const val = datasetSelect.value;
    customWrapper.classList.toggle("hidden", val !== "custom");
    const preset = DATASET_DEFAULTS[val];
    if (preset) {
      document.getElementById("min-support").value = preset.min_support;
      document.getElementById("min-threshold").value = preset.min_threshold;
      document.getElementById("metric-select").value = preset.metric;
      document.getElementById("basket-input").value = preset.basket;
    }
  });

  document.getElementById("mine-form").addEventListener("submit", (evt) => {
    evt.preventDefault();
    runMining();
  });

  try {
    const res = await fetch("/api/overview");
    if (res.ok) {
      renderOverview(await res.json());
    }
  } catch {
    // Non-fatal; KPI strip displays baseline defaults
  }

  // Run initial live mining pass on Groceries with default basket query
  document.getElementById("basket-input").value = DATASET_DEFAULTS.groceries.basket;
  await runMining();
}

init();
