# Inference Arbiter - Architecture & Build Plan

**Status:** Planning Complete -> Ready for Phase 1 Implementation

**Primary Stack:** Python 3.11+ (Java / TypeScript optional later)

**Target Machine:** Mac Studio (local Ollama / llama.cpp + OpenRouter cloud escalation)

**Companion Blueprint:** `agentic_harness_architecture.html`

## 1. Core Thesis & Goals

This project is a hands-on, transparent **agentic systems harness** for Q&A workflows. Rather than relying on a single monolithic "God LLM" end-to-end-or adopting a black-box router-the harness uses **model chaining** to put specialized models where they perform best.

### Primary Objectives

1. **Best at Their Best (Cost & Speed):** Route routine triage, query rewriting, extraction, and simple Q&A to fast local models (`Ollama` / `llama.cpp`) at \$0 marginal API cost, reserving mid-tier cloud models and frontier models (via `OpenRouter`) strictly for high-ambiguity reasoning or repair.

2. **Defeat "Confident Wrong" Answers:** Enterprise guardrails mostly focus on security/compliance (PII, prompt injection). This harness focuses on **architectural and deterministic guardrails**-strict schema handoffs, groundedness verification, and sandboxed code/math execution-to catch plausible hallucinations before they reach the user.

3. **Zero Black Boxes:** Every routing decision, confidence threshold, fallback path, and token cost is owned by transparent Python code and logged locally for empirical benchmarking.

4. **Progressive Complexity:** Start ruthlessly small (one question in, one answer out, zero autonomous loops) and only introduce verification loops, tools, and MicroVM sandboxes when the workflow earns them.

## 2. Environment & Workspace Lane Separation

To keep concerns clean and prevent tool overlap on the Mac Studio, maintain **three separate lanes**:

| Lane | Tool / Repo | Role | 
| ----- | ----- | ----- | 
| **1. Interactive Coding** | Editor and terminal | Interactive code edits in the working tree. | 
| **2. Heavy Autonomous Agent** | Existing Hermes Agent v0.21.3 (`~/.hermes/hermes-agent` + SerpApi) | Read-only repo audits, `/goal` runs with test gates, `/learn` skill capture, and unattended `/cron` jobs. **Not imported into Phase 1-2.** | 
| **3. Inference Arbiter** | New standalone repo (e.g., `~/work-space/projects/inference-arbiter`) | Lean, transparent Python router, verification pipeline, eval suite, and Phase 3 sandboxed tool runner. | 

### How Hermes Fits (Without Becoming a Dependency)

* **Phase 1-2:** Keep the harness completely independent of Hermes Agent. Optionally skim two files in the local `hermes-260917` checkout for reference patterns only:

  * `agent/auxiliary_client.py` (isolating cheap side-task models from the main reasoning model)

  * `hermes_cli/config.py` (separating `config.yaml` from `.env` secrets)

* **Local Specialist Option:** Open-weight **Hermes 3 (8B)** or **Hermes 4/4.3** models can be run locally via Ollama for strong instruction-following and structured JSON extraction.

* **Phase 3 Optional Bridge:** If the harness exposes a thin local OpenAI-compatible endpoint (`http://127.0.0.1:8000/v1`), Hermes Agent can point at it (`--provider custom --base-url ...`)-making *your transparent router* the brain behind Hermes, rather than the other way around.

## 3. Design Principles & Lessons Adopted

### Borrowed from Jev Router Architecture (Without Adopting a Black Box)

* **Treat prompt cache as a routing variable:** Naive per-turn switching destroys warm prompt caches and forces the new model to re-read the entire conversation. In multi-turn sessions, prefer keeping the active model and adjusting **reasoning effort** up or down; only switch models when expected accuracy gain outweighs cache-loss cost.

* **Client-side fallback for fail-closed routers:** Fail-closed routing is good (never route blind), but any triage/router call must sit inside a Python `try/except` with a strict timeout and an explicit fallback tier (e.g., local Ollama or a fixed mid-tier cloud model).

* **Log the *why*, not just the *what*:** Record `tier`, `model`, `effort`, `confidence`, `route_reason`, `latency_ms`, and `cost_usd` in SQLite so Phase 2 evals can diagnose *why* a wrong answer happened.

* **Text-only triage:** Strip attachments, images, and bulky retrieved documents before sending prompts to the triage classifier.

* **Hybrid cloud option:** Because `typesafe/jev-router` has an undisclosed candidate model pool and cannot route to local models, never let it be the top-level controller. If tested at all, treat it strictly as one swappable target under the Cloud branch in `models.yaml`.

### Borrowed from Local Hermes Setup & Usage (`HERMES_SETUP.md`, `USAGE.md`)

* **Never trust snippet-only search answers:** Search snippets produce plausible, confident reconstructions with broken details (e.g., non-chronological EMVCo bulletin lists that were only corrected after fetching the actual source page). Any web-backed Q&A flow in Phase 2+ must include a deterministic page fetch/extract step before grounding.

* **Distill raw text locally before cloud calls:** Raw HTML/page fetches burn cloud context and money. Use a local Ollama model to strip boilerplate and extract clean facts before escalating to paid cloud models.

* **Strict secrets hygiene:** Keep all API keys strictly in `.env` (never in prompts, logs, SQLite tables, or chat transcripts). Block tool access to `.env` in the new repo.

## 4. Three-Phase Build Roadmap

### Phase 1: Bare-Metal Router (Local + Cloud)

**Goal:** One question in, one answer out, zero autonomous loops. Prove clean routing, config-driven model swaps, and telemetry.

* **Core Components:**

  * **Unified Client:** Standard Python `openai` SDK pointing at Ollama (`http://localhost:11434/v1`) and OpenRouter (`https://openrouter.ai/api/v1`).

  * **Config-Driven Tiers (`models.yaml`):** Map `local`, `mid`, and `frontier` tiers to model IDs so benchmarking a new model requires zero code changes.

  * **Fast Triage + Deterministic Router:** Lightweight classifier (local small model or Jev) returns a route label and confidence score; plain Python `if/else` thresholds enforce automatic escalation (e.g., `confidence < 0.80` -> stronger tier) plus timeout fallback.

  * **Async Telemetry (`SQLite` / `JSONL`):** Log every run's `trace_id`, prompt hash/text, triage decision, route reason, tier, model, latency, and estimated token cost.

  * **Optional Thin Proxy:** Wrap the router in a minimal FastAPI `/v1/chat/completions` endpoint so both CLI scripts and external clients (like Hermes) can call it.

* **Exit Criteria:** Can answer queries across Local, Mid, and Frontier tiers from a single entry point, survive a simulated router timeout via fallback, and query cost/latency breakdowns in SQLite.

### Phase 2: Verification & Evals (Tackling "Confident Wrong")

**Goal:** Establish an empirical baseline and catch hallucinations/schema drift before adding tools or autonomy.

* **Core Components:**

  * **Context Prep & Rewrite (Local Ollama):** Clean noisy inputs or fetched pages and emit structured JSON facts.

  * **Pydantic v2 Schema Gate:** Validate intermediate LLM outputs against strict Pydantic models. Catch `ValidationError` in Python and automatically escalate that step to a stronger tier so malformed outputs never poison downstream steps.

  * **Groundedness Verifier + Bounded Repair:** Check whether draft claims are backed by retrieved/extracted sources. If ungrounded, trigger **at most one** repair loop to the Frontier model with the specific critique.

  * **Offline Eval Harness (30-50 Golden Questions):** Replay a fixed test set after every prompt, threshold, or model change and score the multi-model chain against a **single direct mid-tier cloud call** (accuracy, cost, p95 latency).

* **Exit Criteria:** Hard scorecard data proving the multi-model chain beats (or matches at significantly lower cost) a single direct mid-tier model call on the 30-50 question suite.

### Phase 3: Tool Execution & Docker Sandboxes (MicroVM Isolation)

**Goal:** Let models prove their work against deterministic runtimes (Python, unit tests, live APIs/docs) without exposing the host Mac Studio.

* **Core Components:**

  * **Host-Side Tool Dispatch Gate:** Validates tool allow-lists, enforces timeouts, strips host credentials, and controls all traffic crossing the VM boundary.

  * **Docker Sandboxes (`sbx` CLI) MicroVMs:** Ephemeral MicroVMs with dedicated Linux kernels, private Docker engines, and strict network allow/deny policies (`--network=none` for pure compute).

  * **Deterministic Verifiers Inside Sandbox:**

    * Python & math runner (eliminates arithmetic/logic hallucinations).

    * Unit test / compiler runner (eliminates "broken code that looks fine").

    * Per-sandbox **scoped MCP servers** (e.g., read-only local docs or specific GitHub repos, never global host access).

  * **Evidence Verifier:** Grounds final answers directly against sandbox exit codes, `stdout`, and structured MCP citations, then destroys the ephemeral MicroVM.

* **Exit Criteria:** End-to-end Q&A flow that verifies code or calculations inside an isolated `sbx` MicroVM with zero host credential exposure and full audit logging in SQLite.

## 5. Companion Reference Files

* `agentic_harness_architecture.html` - Interactive 3-tab architectural flow diagram with node inspector, code snippets, and animated packet simulations.

* `jev-router-architecture.html` - Reference breakdown of Jev Router mechanics, cache-aware effort tuning, fail-closed limits, and payments-stack analogy.

* `HERMES_SETUP.md` / `hermes-agent-guide.md` / `USAGE.md` - Local Mac Studio Hermes v0.21.3 configuration, SerpApi setup, and operational cautions.

## 6. Immediate Next Steps (When Ready to Build Phase 1)

1. **Inventory local Ollama models** on the Mac Studio (e.g., `qwen2.5:7b`, `hermes3:8b`, or similar) to populate `models.yaml`.

2. **Choose Phase 1 entry point:** Start with a pure Python CLI script, or include the thin FastAPI `/v1/chat/completions` wrapper from day one.

3. **Scaffold the Phase 1 repo:** Directory layout, `models.yaml`, `.env.example`, `router.py`, `telemetry.py` (SQLite), and a 5-question smoke test.

4. **Draft the initial 20-30 golden Q&A test cases** (mixing simple lookups, tricky reasoning, and domain-specific questions) so Phase 2 benchmarking is ready as soon as Phase 1 runs.