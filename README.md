# OpenCodeWEB OS — AiA (Unlimited Self-Evolving Master Intelligence)

AiA is the master intelligence of OpenCodeWEB OS: a **Supervisor-Observer-Executor**
engine with **zero token thresholds**, a **never-stopping learning loop**, and a
**privacy-safe federated swarm** that connects every installation worldwide.

- **Repo:** github.com/OpenCodeWEB/AiA
- **Canonical code path:** `/opt/opencode/lib/aia` (env `AIA_LIB_DIR` overrides)
- **Persistent brain:** `/opt/opencode/aia_brain` (env `AIA_BRAIN_DIR` overrides;
  Windows default: `%USERPROFILE%\opencode\aia_brain`)
- **Marketplace hub:** [github.com/marketplace/OpenCodeWEB](https://github.com/marketplace/OpenCodeWEB)
- **Central knowledge swarm:** `https://aia-brain.opencode.workers.dev/v1` (worker in `worker/`)

## Architecture

```
process_task(prompt)
 ├─ 1. recall from vector memory (unlimited context, hierarchical compression)
 ├─ 2. evaluate_native_capability → YES?  → native execution (skill library)
 └─ 3. NO? → delegate to model swarm (Gemini → opencode → mock)
            → observe → assimilate → promote repeated successes to skills
 └─ 4. learn user preferences + ingest into memory (never stops)
```

## Modules

| File | Role |
|---|---|
| `aia_core_engine.py` | Master controller (CLI: `python aia_core_engine.py --help`) |
| `vector_memory.py` | Sliding window + hashing-TF-IDF compression (pure Python, no deps) |
| `learning_loop.py` | Observations, 3× skill promotion, anti-patterns, user profile |
| `github_sync.py` | Daily GitHub trending OSS → pattern extraction |
| `gemini_bridge.py` | Gemini via Google account (gemini-cli OAuth) — **zero API key** |
| `federated_learning_sync.py` | Privacy-safe swarm push + OTA skill patch apply |
| `health_api.py` | Marketplace-facing `/health` + `/status` HTTP API |
| `executors/` | Gemini → opencode → mock delegation registry |
| `worker/` | Central Brain Cloudflare Worker (validation + patch feed) |

## Privacy contract (federated learning)

Raw code, prompts, and personal data **never leave the device**. Uploads contain
only a category, a numeric feature vector of an *abstracted* solution, outcome
stats, and an HMAC signature bound to a per-install secret.

## Quick Start

```bash
pip install pytest
python -m pytest -q

# PRD demo — first run delegates + learns, third run executes natively
python aia_core_engine.py --demo --executor mock

python aia_core_engine.py --status
python aia_core_engine.py --sync-github
python aia_core_engine.py --connect-gemini   # install gemini CLI first
python aia_core_engine.py --swarm-push       # anonymized learnings → hub
python aia_core_engine.py --swarm-pull       # OTA skill patches ← hub
python health_api.py                         # serves /health /status on :8686
```

## Tests

74+ pytest tests cover routing, promotion, vector recall, abstraction privacy,
watermark sync, and patch validation.
