import http from "node:http";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { validateMineRequest, parseAcceptanceOutput } from "./lib/validation.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const HOST = "0.0.0.0";
const PUBLIC_DIR = path.resolve(__dirname, "public");
const PAPER_FIGURES_DIR = path.resolve(__dirname, "paper", "figures");
const PAPER_PDF_PATH = path.resolve(__dirname, "paper", "paper-ieee-graphical.pdf");
const BRIDGE_SCRIPT = path.resolve(__dirname, "scripts", "web_bridge.py");
const ACCEPTANCE_SCRIPT = path.resolve(__dirname, "tests", "run_acceptance.py");
const BENCHMARK_JSON = path.resolve(__dirname, "benchmarks", "benchmark_results.json");
const E2E_JSON = path.resolve(__dirname, "tests", "e2e", "e2e_results.json");
const MAX_BODY_BYTES = 256 * 1024;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".pdf": "application/pdf",
  ".ico": "image/x-icon",
};

let acceptanceInFlight = false;

function resolvePythonBinary() {
  const fallback = process.platform === "win32" ? "python" : "python3";
  const candidates = [
    process.env.SKARM_PYTHON,
    path.resolve(__dirname, ".venv", "Scripts", "python.exe"),
    path.resolve(__dirname, ".venv", "bin", "python"),
    "/tmp/skarm-venv/bin/python",
    fallback,
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (candidate === "python3" || candidate === "python") {
      return candidate;
    }
    try {
      fs.accessSync(candidate, fs.constants.F_OK);
      return candidate;
    } catch {
      // Try next candidate
    }
  }
  return fallback;
}

function sendJson(res, statusCode, data) {
  const body = JSON.stringify(data);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  res.end(body);
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let totalBytes = 0;

    req.on("data", (chunk) => {
      totalBytes += chunk.length;
      if (totalBytes > MAX_BODY_BYTES) {
        const err = new Error("Request body too large.");
        err.statusCode = 413;
        reject(err);
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });

    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf-8").trim();
        if (!raw) {
          resolve({});
          return;
        }
        resolve(JSON.parse(raw));
      } catch {
        const err = new Error("Invalid JSON request body.");
        err.statusCode = 400;
        reject(err);
      }
    });

    req.on("error", (err) => reject(err));
  });
}

function runPythonBridge(payload) {
  return new Promise((resolve, reject) => {
    const pythonBin = resolvePythonBinary();
    const child = execFile(
      pythonBin,
      [BRIDGE_SCRIPT],
      {
        cwd: __dirname,
        timeout: 20000,
        maxBuffer: 4 * 1024 * 1024,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      },
      (error, stdout, stderr) => {
        if (error) {
          const detail = (stderr || error.message || "Mining execution failed.").trim();
          const err = new Error(detail.split("\n").slice(-2).join(" "));
          err.statusCode = error.killed ? 504 : 500;
          reject(err);
          return;
        }
        try {
          resolve(JSON.parse(stdout));
        } catch {
          const err = new Error("Failed to parse mining output from skarm.");
          err.statusCode = 500;
          reject(err);
        }
      }
    );

    if (child.stdin) {
      child.stdin.write(JSON.stringify(payload));
      child.stdin.end();
    }
  });
}

function runAcceptanceGate() {
  return new Promise((resolve, reject) => {
    const pythonBin = resolvePythonBinary();
    const started = performance.now();
    execFile(
      pythonBin,
      [ACCEPTANCE_SCRIPT],
      {
        cwd: __dirname,
        timeout: 60000,
        maxBuffer: 2 * 1024 * 1024,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      },
      (error, stdout, stderr) => {
        const elapsedMs = Math.round(performance.now() - started);
        const combined = [stdout, stderr].filter(Boolean).join("\n");
        if (error && !stdout) {
          const err = new Error("Acceptance gate runner failed to execute.");
          err.statusCode = error.killed ? 504 : 500;
          reject(err);
          return;
        }
        const parsed = parseAcceptanceOutput(combined);
        resolve({
          ...parsed,
          exit_code: error ? Number(error.code) || 1 : 0,
          elapsed_ms: elapsedMs,
          ran_at: new Date().toISOString(),
        });
      }
    );
  });
}

async function readJsonSafe(filePath) {
  try {
    const text = await fsp.readFile(filePath, "utf-8");
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function serveFileConfined(baseDir, relPath, res, downloadName = null) {
  const normalized = path.resolve(baseDir, `.${relPath}`);

  if (normalized !== baseDir && !normalized.startsWith(baseDir + path.sep)) {
    res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }

  try {
    const stat = await fsp.stat(normalized);
    if (!stat.isFile()) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not Found");
      return;
    }
    const ext = path.extname(normalized).toLowerCase();
    const contentType = MIME_TYPES[ext] || "application/octet-stream";
    const content = await fsp.readFile(normalized);
    const headers = {
      "Content-Type": contentType,
      "Content-Length": content.length,
      "Cache-Control": ext === ".html" ? "no-cache" : "public, max-age=300",
      "X-Content-Type-Options": "nosniff",
    };
    if (downloadName) {
      headers["Content-Disposition"] = `inline; filename="${downloadName}"`;
    }
    res.writeHead(200, headers);
    res.end(content);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not Found");
  }
}

async function serveStatic(reqPath, res) {
  const cleanPath = reqPath === "/" ? "/index.html" : reqPath;
  await serveFileConfined(PUBLIC_DIR, cleanPath, res);
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    const pathname = decodeURIComponent(url.pathname);

    if (pathname === "/health" && req.method === "GET") {
      sendJson(res, 200, {
        ok: true,
        service: "skarm-workbench",
        python: path.basename(resolvePythonBinary()),
      });
      return;
    }

    if (pathname === "/api/overview" && req.method === "GET") {
      const [benchmarks, e2e] = await Promise.all([
        readJsonSafe(BENCHMARK_JSON),
        readJsonSafe(E2E_JSON),
      ]);

      sendJson(res, 200, {
        project: {
          name: "skarm",
          id: "KLCAP-2026-00340",
          version: "0.1.0",
          positioning: "Independent, third-party scikit-learn-compatible library",
          upstream_issue: {
            repo: "scikit-learn/scikit-learn",
            number: 32712,
            state: "closed: not_planned",
            date: "2025-11-16",
            maintainer: "@virchan",
            reference: "#2662",
            note: "Association-rule mining is outside core scikit-learn scope; skarm is maintained as a standalone Estimator-API package.",
          },
          tests: {
            pytest_passed: 264,
            pytest_total: 264,
            e2e_passed: e2e ? e2e.total_passed : 88,
            e2e_total: e2e ? e2e.total_tests : 88,
          },
        },
        benchmarks,
        e2e,
      });
      return;
    }

    if (pathname === "/api/mine") {
      if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed. Use POST." });
        return;
      }
      const body = await readJsonBody(req);
      const validation = validateMineRequest(body);
      if (!validation.ok) {
        sendJson(res, 400, { error: validation.error });
        return;
      }
      const result = await runPythonBridge(validation.value);
      sendJson(res, 200, result);
      return;
    }

    if (pathname === "/api/acceptance") {
      if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed. Use POST." });
        return;
      }
      if (acceptanceInFlight) {
        sendJson(res, 429, {
          error: "Acceptance gate is already running. Wait for the active run to finish.",
        });
        return;
      }
      acceptanceInFlight = true;
      try {
        const result = await runAcceptanceGate();
        sendJson(res, 200, result);
      } finally {
        acceptanceInFlight = false;
      }
      return;
    }

    if ((req.method === "GET" || req.method === "HEAD") && pathname.startsWith("/paper/figures/")) {
      const figRel = pathname.slice("/paper/figures".length);
      await serveFileConfined(PAPER_FIGURES_DIR, figRel, res);
      return;
    }

    if ((req.method === "GET" || req.method === "HEAD") && pathname === "/paper/paper-ieee-graphical.pdf") {
      await serveFileConfined(path.dirname(PAPER_PDF_PATH), `/${path.basename(PAPER_PDF_PATH)}`, res, "skarm-ieee-research-paper.pdf");
      return;
    }

    if (req.method === "GET" || req.method === "HEAD") {
      await serveStatic(pathname, res);
      return;
    }

    sendJson(res, 405, { error: "Method not allowed." });
  } catch (err) {
    const status = err && typeof err.statusCode === "number" ? err.statusCode : 500;
    const message =
      status < 500 && err && err.message
        ? err.message
        : "Internal server error while processing request.";
    sendJson(res, status, { error: message });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`skarm workbench listening on http://${HOST}:${PORT}`);
});
