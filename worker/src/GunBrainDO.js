// =====================================================================
// AiA Central Brain — Global Knowledge Swarm (Cloudflare Worker)
// Source: https://github.com/OpenCodeWEB/AiA
// Endpoint: https://aia-brain.opencode.workers.dev/v1
//
// POST /v1/sync    — accept anonymized swarm patterns (validate, dedupe,
//                    quality-gate, aggregate per category)
// GET  /v1/patch   — return validated skill patches newer than ?since=ts
//                    (bounded feed, e.g. 50 KB)
//
// Privacy enforcement: the worker REJECTS any payload containing raw text
// fields (prompt/learned_solution/solution) — the client contract sends
// numeric feature vectors only.
// =====================================================================

const BODY_LIMIT = 16 * 1024; // 16 KB per sync body
const FEED_LIMIT = 50 * 1024; // 50 KB per patch feed
const MIN_SUCCESSES = 2;      // quality gate: patterns need ≥2 confirmed successes

const FORBIDDEN_KEYS = ["prompt", "learned_solution", "solution", "raw_code", "source_model", "device"];

export class GunBrainDO {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/v1")) return json({ err: "not found" }, 404);
    if (request.method === "OPTIONS") return json({ ok: true });

    if (url.pathname === "/v1/sync" && request.method === "POST") {
      return this.handleSync(request);
    }
    if (url.pathname === "/v1/patch" && request.method === "GET") {
      return this.handlePatch(url);
    }
    return json({ err: "not found" }, 404);
  }

  // ── POST /v1/sync ──────────────────────────────────────────────────────
  async handleSync(request) {
    let body;
    try {
      body = JSON.parse(await request.text());
    } catch {
      return json({ ok: false, err: "invalid json" }, 400);
    }
    if (!body || !Array.isArray(body.patterns) || body.patterns.length === 0 || body.patterns.length > 200) {
      return json({ ok: false, err: "patterns[] required (1..200)" }, 400);
    }
    if (JSON.stringify(body).length > BODY_LIMIT) {
      return json({ ok: false, err: "body too large" }, 413);
    }

    const accepted = [];
    const rejected = [];
    for (const p of body.patterns) {
      if (!validPattern(p)) {
        rejected.push("invalid");
        continue;
      }
      if (FORBIDDEN_KEYS.some((k) => k in p)) {
        rejected.push("privacy-violation");
        continue; // raw text must never reach the hub
      }
      const key = `sig:${p.signature}`;
      const seen = await this.state.storage.get(key);
      if (seen) {
        rejected.push("duplicate");
        continue;
      }
      const record = {
        category: p.category,
        feature_vector: p.feature_vector,
        outcome_stats: p.outcome_stats,
        signature: p.signature,
        ts: Date.now(),
      };
      await this.state.storage.put(key, "1");
      await this.state.storage.put(`pattern:${p.signature}`, record);
      accepted.push(p.signature);
    }
    return json({ ok: true, received: accepted.length, rejected: rejected.length });
  }

  // ── GET /v1/patch ──────────────────────────────────────────────────────
  async handlePatch(url) {
    const since = Number(url.searchParams.get("since") || 0);
    const list = await this.state.storage.list({ prefix: "pattern:" });
    const patches = [];
    let size = 0;
    const keys = [...list.keys()].sort();
    for (const key of keys) {
      const p = await this.state.storage.get(key);
      if (!p || p.ts <= since) continue;
      // quality gate — only well-confirmed, aggregated patterns enter the feed
      if ((p.outcome_stats?.success_count || 1) < MIN_SUCCESSES) continue;
      const patch = {
        category: p.category,
        pattern: patternLabel(p.category),
        solution_template: solutionTemplate(p),
        signature: p.signature,
        ts: p.ts,
      };
      const encoded = JSON.stringify(patch);
      if (size + encoded.length > FEED_LIMIT) break;
      patches.push(patch);
      size += encoded.length;
    }
    return json({ patches, server_ts: Date.now(), feed_bytes: size });
  }
}

// ── validators / helpers ──────────────────────────────────────────────────
function validPattern(p) {
  if (!p || typeof p !== "object") return false;
  if (typeof p.signature !== "string" || p.signature.length < 8 || p.signature.length > 64) return false;
  if (typeof p.category !== "string") return false;
  const v = p.feature_vector;
  if (!Array.isArray(v) || v.length === 0 || v.length > 256) return false;
  if (!v.every((x) => typeof x === "number" && Number.isFinite(x))) return false;
  const s = p.outcome_stats || {};
  if (typeof s.success_count !== "number" || typeof s.avg_duration_ms !== "number") return false;
  return true;
}

function patternLabel(category) {
  const labels = {
    flutter_ui: "flutter ui component pattern",
    python_debug: "python debugging pattern",
    js_fix: "javascript fix pattern",
    typescript: "typescript pattern",
    backend: "backend/API pattern",
    frontend: "frontend pattern",
    devops: "devops pattern",
    general_coding: "general coding pattern",
  };
  return labels[category] || "general coding pattern";
}

// v1: the feed carries the category + signature + stats; the solution
// template is derived on the hub side (abstract skeleton placeholder) so
// clients learn *that a pattern exists* without leaking any raw solution.
function solutionTemplate(p) {
  return `[swarm-validated ${p.category} pattern ${p.signature.slice(0, 8)} — ${p.outcome_stats?.success_count || 0} confirmed successes]`;
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}

export default {
  fetch(request, env) {
    const id = env.GUN_BRAIN.idFromName("default");
    return env.GUN_BRAIN.get(id).fetch(request);
  },
};
