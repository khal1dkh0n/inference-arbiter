# Inference Arbiter - Thread Discussion Summary

A complete synthesis of the discussion moving from a critique of monolithic LLMs to designing a transparent, multi-model **Inference Arbiter** tailored to a local Mac Studio setup.

---

## 1. The Catalyst: Critique of the "LLMs Are the Worst Kind of AI" Premise

* **What the Medium post got right:** It accurately nailed the core production pain point-LLM mistakes are *plausible* rather than obvious ("broken code that looks fine," invented facts buried in convincing explanations), leading to an erosion of trust and "endless forensic investigations."
* **The bait-and-switch:** The headline promised a "Replacement" for LLMs, while the subtitle immediately walked it back to *"part of a bigger AI."* LLMs aren't being replaced; they are being wrapped in orchestration, routing, and guardrails.
* **Reality check:** Moving from a single prompt to larger agentic systems is not a silver bullet-it trades single-prompt overload for new engineering failure modes (compounding errors, higher latency, task drift, and harder debugging).

---

## 2. LLM Chaining, Relocating the "God Model," and the Guardrails Gap

### Static Chains vs. Agentic Workflows
* **LLM Chaining (Prompt Chaining):** Breaking a complex task into smaller, focused calls where each step has a single job and passes structured output downstream (often with deterministic code or database lookups in between).
* **Static Chains:** Hardcoded flow controlled by developer code (`if/else`). Predictable, auditable, and easy to debug, but rigid.
* **Agentic Workflows:** The LLM decides the next step in a loop, invokes tools, and self-corrects until done. More flexible, but prone to task drift and error propagation.

### Relocating the "God Model"
* Monolithic frontier models aren't disappearing-they are being **relocated** in the chain depending on where ambiguity lives:
  * **Front:** Orchestrator that decomposes a messy goal into a clean plan.
  * **Middle:** Escalation path triggered only when lightweight models hit low confidence or complex logic.
  * **End:** Synthesizer, repair agent, or "LLM-as-a-judge" verifier.

### The Guardrails Gap ("Confident Wrong" vs. Enterprise Security)
* Commercial guardrails overwhelmingly focus on **enterprise security and compliance** (PII redaction, prompt-injection defense, toxicity, RBAC) because of regulatory mandates and technical feasibility.
* Catching **"confident wrong" answers** (plausible hallucinations) is fundamentally harder because truth requires a reference point.
* **Key Takeaway:** Because no universal linguistic "truth detector" exists, the most reliable guardrails against confident mistakes are **architectural and deterministic**-forcing models to prove their work against compilers, unit tests, Python execution, live database queries, and strict schema validation.

---

## 3. Project Vision: An Inference Arbiter

* **Philosophy:** Use model chaining not only to cut frontier API costs, but to deploy **"the best at their best"** (specialized, lightweight local models alongside selective cloud escalation) while avoiding opaque black boxes.
* **Language Choice:** Start in **Python** (`openai` SDK works uniformly for Ollama and OpenRouter; `pydantic` enforces strict JSON handoffs). Java or TypeScript can be introduced later if needed.
* **Progressive 3-Phase Architecture** *(visualized in `inference_arbiter_architectural_flow.html` and detailed in `inference_arbiter_architecture_build_plan.md`)*:
  1. **Phase 1 - Bare-Metal Router (Local + Cloud):** One question in, one answer out, zero autonomous loops. Config-driven `models.yaml`, fast triage + deterministic Python `if/else` thresholds, and async SQLite/JSONL telemetry.
  2. **Phase 2 - Verification & Evals:** Local context prep/rewriting, **Pydantic v2 schema gates** (catch `ValidationError` -> escalate step), groundedness verification with a **bounded single Frontier repair loop**, and an offline **30-50 golden question eval harness** comparing the chain against a single direct mid-tier cloud baseline.
  3. **Phase 3 - Tool Execution & Docker Sandboxes:** Host-side tool dispatch gate crossing into ephemeral **Docker Sandboxes (`sbx` CLI) MicroVMs** (dedicated Linux kernel, private Docker engine, network allow/deny, zero host credentials) running deterministic Python/unit-test verifiers and per-sandbox **scoped MCP servers**.

---

## 4. Takeaways from Jev Router Architecture (`jev-router-architecture.html`)

* **Mental Model (Payments Analogy):** OpenRouter acts as the *payment gateway/orchestrator* (owns all connections), Jev acts as the *routing rules engine* (advises without transacting), and the chosen LLM acts as the *acquirer/processor*.
* **4 Patterns Adopted for Our Harness:**
  1. **Treat warm prompt cache as a routing asset:** Rather than switching models every turn (which throws away cached context), default to keeping the active model and tuning **reasoning effort** up or down; only switch models when expected accuracy gain exceeds cache-loss cost.
  2. **Client-side fallback around fail-closed routers:** Fail-closed routing avoids routing blind, but the client must wrap triage calls in a Python `try/except` with a strict timeout and an explicit fallback tier.
  3. **Log the *why*, not just the *what*:** Store `effort`, `confidence`, and `route_reason` in SQLite alongside latency and token cost for Phase 2 diagnostics.
  4. **Text-only triage:** Strip attachments and bulky documents before running routing classification.
* **Caution / Why Not Make It the Top-Level Controller:** `typesafe/jev-router` is a cloud-only black box (undisclosed candidate model pool, unavailable on regional `us`/`eu` domains, cannot route to local Ollama). Keep top-level routing in transparent Python, and treat `jev-router` strictly as an optional swappable cloud target if ever tested.

---

## 5. Evaluating Hermes Agent (`HERMES_SETUP.md`, `hermes-agent-guide.md`, `USAGE.md`)

* **Current Mac Studio Setup:** Hermes Agent v0.21.3 (`~/.hermes/hermes-agent`) with 18 tools, 89 skills, and SerpApi (Google Light, 250 searches/month free tier).
* **Verdict:** **Do not include Hermes inside Phase 1-2.** Hermes (~700K lines of Python) is a heavy autonomous agent designed for repo audits, `/goal` loops, skill capture, and `/cron` jobs-importing it would obscure the routing mechanics this project is meant to master.
* **Three-Lane Workspace Separation:**
  1. **Editor and terminal:** Interactive code edits in the working tree.
  2. **Hermes Agent:** Read-only analysis, research, `/learn` skill capture, and unattended jobs.
  3. **Inference Arbiter:** Standalone lean Python repo (e.g., `~/work-space/projects/inference-arbiter`).
* **3 Operational Lessons Pulled from Hermes Docs:**
  * **Snippet-only search traps:** In the EMVCo test, SerpApi snippets produced a confident, non-chronological bulletin list that was only corrected after fetching the actual Q2 2026 newsletter page. Web-backed Q&A must include a deterministic fetch/extract step before grounding.
  * **Local distillation before cloud calls:** Use a small local Ollama model to strip boilerplate from raw fetched HTML/text before sending it to paid cloud models.
  * **Secrets hygiene:** Keep keys strictly in `.env` (never in chat, logs, or SQLite) and block tool access to `.env`.
* **Optional Future Touchpoints:**
  * Reference patterns only (not imports) in local `hermes-260917`: `agent/auxiliary_client.py` and `hermes_cli/config.py`.
  * **Phase 3 Reverse Bridge:** Expose the custom Python router via a thin local FastAPI `/v1/chat/completions` proxy and point Hermes at it (`--provider custom --base-url http://127.0.0.1:8000/v1`), making *your transparent router* the brain behind Hermes.

---

## 6. Mac Studio Local Ollama Inventory & Tier Mapping

No new model downloads are needed-the existing `ollama list` covers every tier of the harness:

| Harness Tier | Selected Model | Footprint | Role |
| :--- | :--- | :--- | :--- |
| **Triage / Router** | `llama3.2:3b` *(or `qwen3:14b`)* | 2.0 GB *(9.3 GB)* | Permanently warm sub-second routing (`qwen3:14b` if 3B drops strict JSON). |
| **Local Workhorse & Prep** | `qwen3-14b-ctx40k:latest` | 9.3 GB | Everyday Q&A, query rewriting, and distilling raw fetched text. |
| **Code Specialist** | `qwen2.5-coder:32b` | 19 GB | Dedicated local lane for code, SQL, and technical debugging. |
| **Local Heavy / Reasoning** | `gpt-oss-120b-ctx16k:latest` | 65 GB | 117B MoE (~5.1B active) with configurable reasoning effort (`low`/`medium`/`high`)-allows testing "keep model, tune effort" locally. |
| **Cloud Escalation** | `deepseek-v4-flash:cloud`, `kimi-k3.1-pro:cloud` *(+ OpenRouter)* | Cloud | Already routed via Ollama's `:cloud` ZDR endpoint; OpenRouter adds frontier models. |
| **Phase 2 Embeddings** | `nomic-embed-text:latest` | 274 MB | Local RAG and groundedness retrieval. |
| **Eval Challengers** | `llama3.3-70b-ctx64k:latest`, `deepseek-r1:70b`, `phi4:latest` | 9-42 GB | Reserved for Phase 2 offline benchmark comparisons. |

### Mac Studio Hardware Gotchas
* **Avoid Unified Memory swap thrashing:** Don't chain two massive local models (e.g., 42 GB -> 65 GB) back-to-back, which forces 10-30s disk reloads. Keep the triage model small/resident and log Ollama's `load_duration` in SQLite.
* **Always pin `-ctx` tags:** Use `qwen3-14b-ctx40k:latest` and `gpt-oss-120b-ctx16k:latest` in `models.yaml` instead of bare tags so Ollama doesn't silently truncate longer prompts.

---

## 7. Artifacts Produced & Next Decision Point

* `inference_arbiter_architectural_flow.html` - Interactive 3-tab architectural flow diagram with clickable node inspector, code snippets, and animated packet simulations.
* `inference_arbiter_architecture_build_plan.md` - Actionable architecture & 3-phase build plan with exit criteria.
* `inference_arbiter_thread_discussion_summary.md` - This full-thread discussion recap.
* **Next Step:** Scaffold the **Phase 1 Python starter** (`models.yaml`, router with fail-closed fallback, Ollama/OpenRouter client, and SQLite telemetry)-choosing between a **plain CLI script** first or including the **thin FastAPI `/v1/chat/completions` proxy** from day one.