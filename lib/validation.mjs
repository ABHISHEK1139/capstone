export const ALLOWED_DATASETS = new Set([
  "demo",
  "groceries",
  "market_basket",
  "tic_tac_toe",
  "mushroom",
  "custom",
]);

export const ALLOWED_ALGORITHMS = new Set([
  "auto",
  "apriori",
  "fpgrowth",
  "eclat",
]);

export const ALLOWED_METRICS = new Set([
  "confidence",
  "lift",
  "support",
  "leverage",
  "conviction",
  "zhangs_metric",
]);

export const MAX_CUSTOM_TRANSACTIONS = 2000;
export const MAX_ITEMS_PER_TRANSACTION = 128;
export const MAX_ITEM_LENGTH = 64;

/**
 * Parses comma-separated transaction lines into normalized string arrays.
 * @param {string} rawText
 * @returns {string[][]}
 */
export function parseTransactionsText(rawText) {
  if (typeof rawText !== "string") {
    return [];
  }
  const lines = rawText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const transactions = [];
  for (const line of lines) {
    const items = line
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    if (items.length > 0) {
      transactions.push(items);
    }
  }
  return transactions;
}

/**
 * Parses a comma-separated basket query string into unique trimmed item names.
 * @param {string} rawBasket
 * @returns {string[]}
 */
export function parseBasketQuery(rawBasket) {
  if (typeof rawBasket !== "string") {
    return [];
  }
  const seen = new Set();
  const items = [];
  for (const part of rawBasket.split(",")) {
    const trimmed = part.trim();
    if (trimmed.length > 0 && trimmed.length <= MAX_ITEM_LENGTH && !seen.has(trimmed)) {
      seen.add(trimmed);
      items.push(trimmed);
    }
  }
  return items.slice(0, 32);
}

/**
 * Validates and normalizes a POST /api/mine payload.
 * Never silently substitutes fallbacks for invalid required parameters.
 * @param {unknown} body
 * @returns {{ ok: true, value: object } | { ok: false, error: string }}
 */
export function validateMineRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return { ok: false, error: "Request body must be a JSON object." };
  }

  const dataset = typeof body.dataset === "string" ? body.dataset.trim() : "";
  if (!ALLOWED_DATASETS.has(dataset)) {
    return {
      ok: false,
      error: `Invalid dataset. Expected one of: ${Array.from(ALLOWED_DATASETS).join(", ")}.`,
    };
  }

  const algorithm = typeof body.algorithm === "string" ? body.algorithm.trim() : "auto";
  if (!ALLOWED_ALGORITHMS.has(algorithm)) {
    return {
      ok: false,
      error: `Invalid algorithm. Expected one of: ${Array.from(ALLOWED_ALGORITHMS).join(", ")}.`,
    };
  }

  const metric = typeof body.metric === "string" ? body.metric.trim() : "confidence";
  if (!ALLOWED_METRICS.has(metric)) {
    return {
      ok: false,
      error: `Invalid metric. Expected one of: ${Array.from(ALLOWED_METRICS).join(", ")}.`,
    };
  }

  const minSupport = Number(body.min_support);
  if (!Number.isFinite(minSupport) || minSupport <= 0 || minSupport > 1) {
    return {
      ok: false,
      error: "min_support must be a finite number in the interval (0, 1].",
    };
  }

  const minThreshold = Number(body.min_threshold);
  if (!Number.isFinite(minThreshold) || minThreshold < -1 || minThreshold > 1000) {
    return {
      ok: false,
      error: "min_threshold must be a finite number between -1 and 1000.",
    };
  }

  const maxCandidatesRaw = body.max_candidates === undefined || body.max_candidates === null || body.max_candidates === ""
    ? 100000
    : Number(body.max_candidates);
  if (!Number.isInteger(maxCandidatesRaw) || maxCandidatesRaw < 1 || maxCandidatesRaw > 1000000) {
    return {
      ok: false,
      error: "max_candidates must be an integer between 1 and 1,000,000.",
    };
  }

  const topNRaw = body.top_n === undefined || body.top_n === null ? 50 : Number(body.top_n);
  if (!Number.isInteger(topNRaw) || topNRaw < 1 || topNRaw > 200) {
    return {
      ok: false,
      error: "top_n must be an integer between 1 and 200.",
    };
  }

  let transactions = null;
  if (dataset === "custom") {
    if (typeof body.transactions_text === "string") {
      transactions = parseTransactionsText(body.transactions_text);
    } else if (Array.isArray(body.transactions)) {
      transactions = body.transactions
        .filter((row) => Array.isArray(row))
        .map((row) =>
          row
            .map((item) => (typeof item === "string" ? item.trim() : ""))
            .filter((item) => item.length > 0)
        )
        .filter((row) => row.length > 0);
    } else {
      return {
        ok: false,
        error: "Custom dataset requires non-empty transactions_text or transactions array.",
      };
    }

    if (transactions.length < 2) {
      return {
        ok: false,
        error: "Provide at least 2 non-empty custom transactions (one transaction per line).",
      };
    }
    if (transactions.length > MAX_CUSTOM_TRANSACTIONS) {
      return {
        ok: false,
        error: `Custom dataset exceeds limit of ${MAX_CUSTOM_TRANSACTIONS} transactions.`,
      };
    }
    for (const row of transactions) {
      if (row.length > MAX_ITEMS_PER_TRANSACTION) {
        return {
          ok: false,
          error: `Each transaction may contain at most ${MAX_ITEMS_PER_TRANSACTION} items.`,
        };
      }
      for (const item of row) {
        if (item.length > MAX_ITEM_LENGTH) {
          return {
            ok: false,
            error: `Item labels must be ${MAX_ITEM_LENGTH} characters or fewer.`,
          };
        }
      }
    }
  }

  const basket = Array.isArray(body.basket)
    ? body.basket
        .map((item) => (typeof item === "string" ? item.trim() : ""))
        .filter((item) => item.length > 0 && item.length <= MAX_ITEM_LENGTH)
        .slice(0, 32)
    : parseBasketQuery(typeof body.basket_text === "string" ? body.basket_text : "");

  return {
    ok: true,
    value: {
      dataset,
      algorithm,
      metric,
      min_support: minSupport,
      min_threshold: minThreshold,
      max_candidates: maxCandidatesRaw,
      top_n: topNRaw,
      transactions,
      basket,
      compare_engines: Boolean(body.compare_engines),
    },
  };
}

/**
 * Parses stdout of tests/run_acceptance.py into structured criterion rows and summary metrics.
 * @param {string} output
 */
export function parseAcceptanceOutput(output) {
  const lines = String(output || "").split(/\r?\n/);
  const checks = [];
  let verdict = "UNKNOWN";
  let ac4Summary = null;

  for (const line of lines) {
    const trimmed = line.trim();

    // Scorecard summary lines: "* KPI-1 (Precision vs Oracle): Apriori 100.0%, FP 100.0%, Eclat 100.0% [PASS]"
    const scoreMatch = trimmed.match(/^\*\s+([^:]+):\s+(.*?)\s+\[(PASS|FAIL)\]$/);
    if (scoreMatch) {
      checks.push({
        id: scoreMatch[1].trim(),
        detail: scoreMatch[2].trim(),
        status: scoreMatch[3],
      });
      if (trimmed.includes("KPI-4 (Strict p95")) {
        ac4Summary = trimmed.replace(/^\*\s+/, "");
      }
      continue;
    }

    // Individual negative test lines: "-> NT-1: Candidate explosion detection & safe degradation [PASS]"
    const ntMatch = trimmed.match(/^->\s+(NT-\d+):\s+(.*?)\s+\[(PASS|FAIL)\]$/);
    if (ntMatch) {
      checks.push({
        id: ntMatch[1].trim(),
        detail: ntMatch[2].trim(),
        status: ntMatch[3],
      });
      continue;
    }

    // Legacy / generic "[PASS] AC-1: detail" fallback
    const legacyMatch = trimmed.match(/^\[(PASS|FAIL)\]\s+([A-Z]+-\d+):\s+(.*)$/);
    if (legacyMatch) {
      checks.push({
        status: legacyMatch[1],
        id: legacyMatch[2],
        detail: legacyMatch[3],
      });
      continue;
    }

    if (trimmed.startsWith("FINAL GATE VERDICT:")) {
      verdict = trimmed.replace("FINAL GATE VERDICT:", "").trim();
    }
    if (trimmed.includes("KPI-4 (Strict p95")) {
      ac4Summary = trimmed.replace(/^\*\s+/, "");
    }
  }

  const passedCount = checks.filter((c) => c.status === "PASS").length;
  const failedCount = checks.filter((c) => c.status === "FAIL").length;

  return {
    verdict,
    passed_count: passedCount,
    failed_count: failedCount,
    total_count: checks.length,
    ac4_summary: ac4Summary,
    checks,
    raw_output: String(output || "").trim(),
  };
}
