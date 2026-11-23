import test from "node:test";
import assert from "node:assert/strict";
import {
  parseTransactionsText,
  parseBasketQuery,
  validateMineRequest,
  parseAcceptanceOutput,
} from "./validation.mjs";

test("parseTransactionsText splits lines and trims comma-separated items", () => {
  const parsed = parseTransactionsText(" milk, bread , butter \n\n beer, diapers \r\n");
  assert.deepEqual(parsed, [
    ["milk", "bread", "butter"],
    ["beer", "diapers"],
  ]);
});

test("parseBasketQuery deduplicates and trims items", () => {
  assert.deepEqual(parseBasketQuery("milk, bread, milk,  "), ["milk", "bread"]);
});

test("validateMineRequest accepts valid preset payload", () => {
  const result = validateMineRequest({
    dataset: "groceries",
    algorithm: "fpgrowth",
    metric: "lift",
    min_support: 0.02,
    min_threshold: 1.2,
    max_candidates: 50000,
    top_n: 25,
    basket_text: "whole milk, yogurt",
  });
  assert.equal(result.ok, true);
  assert.equal(result.value.dataset, "groceries");
  assert.equal(result.value.algorithm, "fpgrowth");
  assert.deepEqual(result.value.basket, ["whole milk", "yogurt"]);
});

test("validateMineRequest rejects out-of-range min_support without silent fallback", () => {
  const resZero = validateMineRequest({
    dataset: "demo",
    algorithm: "auto",
    metric: "confidence",
    min_support: 0,
    min_threshold: 0.5,
  });
  assert.equal(resZero.ok, false);

  const resHigh = validateMineRequest({
    dataset: "demo",
    algorithm: "auto",
    metric: "confidence",
    min_support: 1.5,
    min_threshold: 0.5,
  });
  assert.equal(resHigh.ok, false);
});

test("validateMineRequest rejects unknown datasets and algorithms", () => {
  const badDs = validateMineRequest({
    dataset: "../etc/passwd",
    algorithm: "auto",
    metric: "confidence",
    min_support: 0.2,
    min_threshold: 0.5,
  });
  assert.equal(badDs.ok, false);

  const badAlgo = validateMineRequest({
    dataset: "demo",
    algorithm: "shell_exec",
    metric: "confidence",
    min_support: 0.2,
    min_threshold: 0.5,
  });
  assert.equal(badAlgo.ok, false);
});

test("validateMineRequest requires at least 2 non-empty transactions for custom dataset", () => {
  const single = validateMineRequest({
    dataset: "custom",
    algorithm: "eclat",
    metric: "confidence",
    min_support: 0.5,
    min_threshold: 0.5,
    transactions_text: "milk, bread",
  });
  assert.equal(single.ok, false);

  const valid = validateMineRequest({
    dataset: "custom",
    algorithm: "eclat",
    metric: "confidence",
    min_support: 0.5,
    min_threshold: 0.5,
    transactions_text: "milk, bread\nmilk, butter\nbread, butter",
  });
  assert.equal(valid.ok, true);
  assert.equal(valid.value.transactions.length, 3);
});

test("parseAcceptanceOutput extracts real scorecard and negative test rows", () => {
  const sample = [
    "  -> NT-1: Candidate explosion detection & safe degradation [PASS]",
    "  * KPI-1 (Precision vs Oracle):       Apriori 100.0%, FP 100.0%, Eclat 100.0% [PASS]",
    "  * KPI-4 (Strict p95 <= 0.80x Ref):   FP 0.451x (2.22x faster), Eclat 0.159x (6.29x faster) [PASS]",
    "FINAL GATE VERDICT: PASSED (ALL STRICT CHARTER TARGETS & GUARDRAILS SATISFIED)",
  ].join("\n");
  const parsed = parseAcceptanceOutput(sample);
  assert.equal(parsed.passed_count, 3);
  assert.equal(parsed.failed_count, 0);
  assert.equal(parsed.verdict, "PASSED (ALL STRICT CHARTER TARGETS & GUARDRAILS SATISFIED)");
  assert.equal(parsed.checks[0].id, "NT-1");
  assert.equal(parsed.checks[2].id, "KPI-4 (Strict p95 <= 0.80x Ref)");
  assert.ok(parsed.ac4_summary.includes("FP 0.451x"));
});
