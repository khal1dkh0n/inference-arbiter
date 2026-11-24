# Inference Arbiter - Router (Phase 1), Verification Chain and Open-Book Pipeline (Phase 2), Sandboxed Execution (Phase 3)

One question in, one answer out, zero autonomous loops. By default the harness picks the
strategy the Phase 2 evals measured best: one mid-tier call for a question without
sources, one call with the sources pasted in when they fit the context window, and a
grounded pipeline with citations when they do not. Plain Python `if/else` makes every
choice and every decision is logged to SQLite. The other strategies stay reachable by
flag: `--route` (a small local model triages the question and picks a tier), `--chain`
(draft, schema gate, verifier, at most one repair by the strongest tier) and
`--require-citations` (always the grounded pipeline: every quote checked in code and
each claim judged against the passage it cites) and `--execute` (the model writes a short
Python program, an isolated `sbx` MicroVM runs it, and code compares the output with the
answer the model claimed, when it claimed one) and `--code-test` (for a question that asks
for a Python function: a solution and unit tests are written by separate calls, the test
writer never sees the solution, both run in one sandbox and the answer says whether the
solution passed its generated tests) and `--docs DIR` (the documents of one local folder are
fetched through a read-only MCP server behind a tool gate that logs every call, and the
grounded pipeline answers from them). Governing specs: `inference_arbiter_architecture_build_plan.md` and
`inference_arbiter_thread_discussion_summary.md`.

## Setup

Requires `uv` and a running Ollama daemon on `http://localhost:11434`.

```bash
uv sync --python 3.12
```

That installs the `mcp` package too (it is in the dev group), which only mcp mode
(`--docs`) uses. An install without the dev group needs `uv sync --extra mcp` for that
mode; every other mode runs without the package.

For the `frontier` tier, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` and
`FRONTIER_MODEL` (the OpenRouter model id to use). Without them the harness still runs: a
frontier call fails with `missing_model` or `missing_api_key` and hops to its `on_error`
tier.

## Usage

```bash
uv run python -m inference_arbiter ask "What is the capital of France?"
uv run python -m inference_arbiter ask "What confidence value triggers escalation?" --sources notes.md https://example.com/page
```

Without a mode flag the harness chooses (auto mode). The rule is one pure function,
`auto.decide_mode` in `src/inference_arbiter/auto.py`:

| Case | What runs | Calls |
| --- | --- | --- |
| No sources | One call to `auto.closed_book_tier` (`mid`). No triage, no chain; the `on_error` fallback hop still applies | 1 |
| No sources, the computation check is on and says yes | Execute mode on `execute.tier` with the two-program cross-check (see "Computation check" below). On in the shipped config | 1 check + 2 to 4 |
| `--sources`, and the estimated prompt fits `auto.stuffed_fill_ratio` (0.8) of the answering tier's `context_window_tokens` | Stuffed: one call with the full text of every source ahead of the question. No citations | 1 |
| `--sources` that do not fit, or a tier without `context_window_tokens` | The open-book pipeline: retrieval, draft with citations, quote check, grounding verifier, capped escalation | up to 5 |
| `--sources` with `--require-citations` | The open-book pipeline whatever the size, for when the answer must be auditable | up to 5 |

The trace shows the choice and why, for example `mode stuffed (auto: sources fit, 10k of
839k tokens)`, and the run row stores both (`mode`, `auto_reason`). Sources are loaded
once and reused by the chosen mode. Token counts are estimates (characters / 4).

### Computation check

On in the shipped config; `--no-computation-check` turns it off for one call and
`auto.computation_check: false` for every call (`--computation-check` turns it on for
one). With it on, an auto request without sources and without `--tier` first makes one
call to `auto.computation_check_tier` (the `local` tier in the shipped config) with
a typed yes/no question and a confidence: can a short standard-library Python program
compute the answer? The model judges, code decides (`computation_check.decide`, then
`auto.decide_mode`):

| Check result | What runs |
| --- | --- |
| yes, confidence at or above `auto.computation_check_threshold` (0.8) | Execute mode on `execute.tier`, with the cross-check when `auto.computation_cross_check` is true (the default), whatever `execute.cross_check` says |
| no | The closed-book call, as without the check |
| yes below the threshold (`low_confidence`) | The closed-book call |
| timeout, client error or output that fails the schema (`failed`) | The closed-book call |

The check is never run for a request with sources or with an explicit `--route`, `--chain`,
`--stuffed`, `--execute` or `--tier`. Its call is in the trace and in telemetry with the
role `check`, and the verdict is part of `auto_reason`, for example `auto: no sources, the
computation check said no, one closed-book call [check on local: no, confidence 0.90]`.

When auto has sent a question to execute mode, `verified`, `disputed` and `one_sided` are
returned exactly as `--cross-check` returns them. When nothing could be executed (the
drafts failed, `sbx` is missing, the sandbox is unavailable, both programs failed after
their repairs), the harness makes the closed-book call it would have made without the
check and returns that answer under a first line that starts `NOT VERIFIED:`. The run then
reports `mode: route`, `execution_status: failed`, the reason in `auto_reason` and
`execution_note`, and keeps the failed calls and sandbox rows in telemetry.

Measured 2026-11-24, free tiers, one run per case (`eval/results/2026-11-24-auto-execute.md`).
The check alone on 62 labelled cases: `local` (qwen3 14b) precision 0.952, recall 0.976,
median 0.87 s; `triage` (llama3.2:3b) precision 0.809, recall 0.927, median 0.24 s, with
lookups and domain questions among its wrong sends, which is why the shipped tier is
`local`. End to end with the check on `local`: the golden set scored 35/36 with and
without the check (17 questions sent to execution, all `verified` and right, none of them
a lookup, domain or trap case; 0.78 s median added to the other 19); the hard set had 25
of 26 sent, 23 `verified` and right, 1 `disputed`, 1 `one_sided` and right, 1 answered
closed-book and right, and no `verified` answer wrong. A question sent to execution took
11.3 s at the median on the golden set against 3.1 s for the plain call. The check stays
off until the owner turns it on.

Why these defaults: on the Phase 2 suites a single `mid` call was more accurate than the
chain at no cost, a stuffed call tied the pipeline whenever the sources fit, and the
pipeline is the only strategy that works when they do not (see Phase 2 status).

Explicit flags win over auto:

| Flag | Effect |
| --- | --- |
| `--route` | Route mode: a triage call picks the tier by the routing rules below, then one answer call. `--triage-tier` and `--simulate triage-timeout` imply it |
| `--chain` | Chain mode: triage, draft, schema gate, verify, at most one repair |
| `--execute` | Execute mode: `execute.tier` writes a Python program and, optionally, the answer it claims, the program runs in a throwaway sandbox, and the answer says what execution did and did not confirm. Needs the `sbx` CLI. Not combinable with `--chain`, `--route`, `--stuffed`, `--sources`, `--triage-tier` or `--verifier-tier`; `--tier` picks the tier that writes the program. Auto mode picks it only when the computation check is on and says yes |
| `--computation-check` / `--no-computation-check` | Auto mode without sources: run, or do not run, the computation check before the closed-book call, whatever `auto.computation_check` says. A usage error with `--route`, `--chain`, `--stuffed`, `--execute`, `--cross-check`, `--sources` or `--tier` |
| `--cross-check` | Execute mode with a second program: written independently from the question alone, run in its own sandbox, and compared with the first in code. `verified` only when both outputs match; `disputed` when they differ; `one_sided` when only one ran. Implies `--execute`. Off unless this flag or `execute.cross_check` turns it on |
| `--cross-check-tier NAME` | The tier that writes the second program (implies `--cross-check`). Default `execute.cross_check_tier`, else the tier that writes the first |
| `--tier NAME` | Answer with this tier, no triage (`local`, `code`, `local_heavy`, `mid`, `frontier`). Alone it is one forced call; with `--chain` it forces the draft tier; with `--sources` it is the stuffed or drafting tier, and its context window is the one checked |
| `--sources PATH_OR_URL [...]` | Answer from these files or http(s) URLs. Put the question before this flag |
| `--docs DIR` | Mcp mode: answer from the `.md` and `.txt` documents of this folder, fetched through a read-only MCP server behind the tool gate (see "MCP mode"). Not combinable with `--sources` or another mode flag |
| `--require-citations` | With `--sources`: always the open-book pipeline. Without `--sources` it is a usage error: there is nothing to cite |
| `--stuffed` | With `--sources`: always one call with the full text of every source in the prompt. Triage picks the tier unless `--tier` is given (this is the eval baseline, unchanged) |
| `--verifier-tier NAME` | With `--chain`: verifier tier to use (`verify`, `verify_jev`); default is `chain.verifier_tier`. With `--sources`: grounding verifier (`ground_jev`, `ground_chat`) when the pipeline runs; default is `openbook.grounding_verifier_tier` |
| `--triage-tier NAME` | Triage tier to classify with; default is `triage_tier` |
| `--effort low\|medium\|high` | Override the tier's effort |
| `--config PATH` | Config file (default `config/models.yaml`) |
| `--db PATH` | Telemetry database (default `data/telemetry.db`) |
| `--simulate triage-timeout\|answer-error\|verifier-error\|draft-invalid\|quote-mismatch\|grounding-error\|tool-refused` | Inject a failure to exercise the fail paths. `verifier-error` and `draft-invalid` need `--chain`; `quote-mismatch` (every quote of the first draft fails the quote check) and `grounding-error` (the grounding verifier fails) need `--docs`, or `--sources` without `--stuffed`, and select the grounded pipeline; `tool-refused` needs `--docs`: the harness asks the tool gate for a tool that is not on its allow-list, the gate refuses and records it, and the run ends without a model call |
| `--json` | Print the full run result as JSON on stdout |

In code, `AskRequest.mode` defaults to `auto` and takes `route`, `chain`, `openbook`,
`stuffed`, `execute`, `codetest` or `mcp` (with `docs_root`) to choose explicitly;
`require_citations=True` is the flag's equivalent.
`RunResult.mode` is always the mode that ran. The eval runner requests its modes
explicitly, so its strategies mean what they meant before auto existed.

The answer goes to stdout and the routing trace to stderr. Exit codes: `0` answered,
`1` run failed, `2` usage or config error.

Other commands:

- `uv run pytest` - unit tests (fake clients, a fake `sbx`, no network, no real database, no
  model). The tests in `tests/test_mcp_session.py` start the real documents server as a
  subprocess on a temporary folder; they are skipped when the `mcp` package is not installed
- `INFERENCE_ARBITER_LIVE_SANDBOX=1 uv run pytest tests/test_sandbox_live.py -v -s` - the opt-in
  live test: runs an isolation probe in a real sandbox (skipped without the flag or `sbx`)
- `uv run python scripts/smoke.py` - five live questions through the router (route mode)
- `uv run python scripts/report.py` - cost, latency, fallback and chain report from the database

## Routing rules

These apply in route mode (`--route`), chain mode and explicit `--stuffed`. Auto mode makes
no triage call: its tier is `auto.closed_book_tier` (`route_reason: auto`), the open-book
drafter (`configured`) or the `--tier` given (`forced`).

First match wins. The result is recorded as `route_reason`.

| Order | Condition | Selected tier | `route_reason` |
| --- | --- | --- | --- |
| 1 | `--tier` given | that tier | `forced` |
| 2 | triage timed out, errored, or returned invalid output | `fallback_tier` | `triage_timeout`, `triage_error`, `triage_invalid` |
| 3 | triage confidence below `confidence_threshold` | one step up per `escalation` | `low_confidence` |
| 4 | otherwise | the triage label's tier | `triage_label` |

- `local_heavy` is never auto-routed; it is reachable only with `--tier local_heavy`.
- A low-confidence `frontier` label has no step up and stays on `frontier`.
- Triage is text-only: only the text of the last user message is sent, images and
  attachments are dropped, and long text is cut to `triage_max_chars` (head plus tail).
- Triage is fail-closed: no retry and no guessed label.

## Fallback

If the answer call fails, the router hops once to the tier's `on_error` target. The hop is
skipped when the tier has no target, or when the target is a cloud tier and
`allow_cloud_fallback` is false. If the hop also fails, the run returns an explicit error
with no answer text. Clients do not retry on their own.

Error types: `timeout`, `connection`, `http_429`, `http_5xx`, `http_4xx`, `empty`,
`invalid_response`, `missing_api_key`, `missing_model`, `client_error`, plus `simulated_timeout` and
`simulated_error` for failures injected with `--simulate`.

## Chain mode

```bash
uv run python -m inference_arbiter ask "How many legs do three spiders have?" --chain
```

Chain mode is opt-in. It is plain control flow with a
fixed ceiling of four model calls after triage: at most two draft attempts, one verifier
call, one repair call.

| Step | What happens | On failure |
| --- | --- | --- |
| 1. Draft | Triage and the routing rules above pick the draft tier. The model must return JSON matching `DraftAnswer` (`answer`, `key_claims`, `assumptions`), requested through the client's JSON-schema support | see step 2 |
| 2. Schema gate | The output is validated with pydantic. Invalid or unparseable output never moves on | Invalid output: one re-run on the `escalation` tier (or `fallback_tier`), `schema_escalated` is set. Failed call: the single `on_error` hop, `fallback_used` is set. Either way there is exactly one second attempt; if it also fails the run fails with an explicit error (`draft_schema_invalid` or the call's error type) |
| 3. Verify | The tier named by `chain.verifier_tier` returns `supported`, `unsupported` or `uncertain` with a critique | Failed call, timeout or invalid verdict follows `chain.on_verifier_error`: `return_unverified` (return the draft, status `unavailable`) or `repair` |
| 4. Repair | On `unsupported` or `uncertain`, `chain.repair_tier` gets the question, the draft and the critique, once. Its output passes the same schema gate. It is not verified again | The draft is returned, `repaired` stays false, `repair_error_type` is set and `final_tier` stays on the tier that wrote the draft. The repair call never takes the `on_error` hop, so a weaker tier's output is never presented as a repair |

What the verifier is and is not. The questions are closed-book: no verifier has sources to
check the draft against. A verdict is a second model's opinion, not grounding. It catches
answers that miss the question, accept a false premise, contradict themselves, or state
something a different model believes is wrong. It does not catch an error both models
share. Grounding against fetched sources is a later step.

Two verifier tiers ship in `config/models.yaml`; both have `role: verifier` and are never
routed to for an answer:

- `verify` - a chat model (`kimi-k3.1-pro:cloud` through Ollama), deliberately a different
  model from the local drafter. It re-derives the answer and returns a `VerifierVerdict`
  (`verdict`, `critique`, `failed_claims`) through the same schema validation.
- `verify_jev` - TypeSafe Jev through OpenRouter's Decisions API. It is asked three typed
  yes/no judgments (does the answer address the question, does it miss a false premise,
  does it contradict its own key claims) and the verdict is decided in code: any settled
  defect is `unsupported`, any probability between 0.3 and 0.7 is `uncertain`, otherwise
  `supported`. Question wording, thresholds and the truth table are together in
  `src/inference_arbiter/verifier.py`. Jev cannot name which claim is wrong, so its critique
  names the failed check only.

Result fields added for chain mode (`RunResult` and the `runs` table): `mode`,
`verification_status` (`supported`, `unsupported`, `uncertain`, `unavailable`, `skipped`),
`verification_critique`, `repaired`, `schema_escalated`, `verifier_tier`. `RunResult` also
carries `repair_error_type`. In route mode `verification_status` is `skipped`. Reading
them together:

- `verification_status` is the verdict on the draft. After a repair the final answer has
  not been verified.
- `status: ok` with `unsupported` or `uncertain` and `repaired: false` means the repair
  failed (or `max_repairs` is 0) and the answer is a draft the verifier did not support.
- `final_tier` is always the tier that wrote the returned answer.

## Open-book mode

```bash
uv run python -m inference_arbiter ask "What confidence value triggers escalation?" \
  --sources inference_arbiter_architecture_build_plan.md inference_arbiter_thread_discussion_summary.md \
  --require-citations
```

This pipeline runs when auto mode finds the sources too large for one call, or when
`--require-citations` asks for it. Sources that fit go to one stuffed call instead.

The answer goes to stdout followed by numbered citations with their quotes; the trace per
step goes to stderr. Plain control flow with a fixed ceiling of five model calls: one
embedding step (passages not yet in the cache are embedded 64 per request and reported as
one call), at most two draft attempts, one grounding verifier call, one escalation call. There is no triage call: the drafter is `openbook.drafter_tier` unless
`--tier` forces one.

| Step | What happens | On failure |
| --- | --- | --- |
| 1. Load | Local text, Markdown and HTML files are read; URLs are fetched with `httpx` (timeout, size cap, text content types only) and cached under `data/cache/pages/`. HTML is converted with the standard library: script, style, navigation and other page chrome are dropped, headings and code are kept | Any source that cannot be loaded fails the run with `source_load_failed` before a model is called |
| 2. Chunk and retrieve | Passages are split on headings and paragraphs (`max_passage_chars`) with stable ids `S<source>-P<passage>`. The best-ranked passages are taken until `retrieval_token_budget` estimated tokens are filled (or the first `top_k` when no budget is set), ranked by `retrieval_ranking`: `hybrid` (default; reciprocal rank fusion of local embeddings and BM25), `embedding` or `lexical`. Embedding vectors are cached under `data/cache/embeddings/` by content hash. A corpus at or under `full_context_token_budget` skips retrieval | A failed embedding request falls back to the lexical ranking; `retrieval_method` and `retrieval_note` record what ran and why |
| 3. Draft | The drafter gets the question and the numbered passages and must return `OpenbookDraft`: `working`, `answerable`, `answer`, `claims[{text, passage_id, quote}]`, each quote copied verbatim from the cited passage | Invalid output: one retry on `escalation_tier` (`schema_escalated`). Failed call: the single `on_error` hop. A second failure fails the run |
| 4. Quote check | In code, no model: every `passage_id` must exist and every quote must appear in its cited passage after whitespace and case normalisation. Quotes under 8 characters fail | The claim is marked unsupported with the reason (`unknown_passage`, `quote_not_in_passage`, `quote_too_short`) |
| 5. Grounding verifier | Judges each claim that passed step 4 against the passage it cites, plus whether the passages answer the question at all. Skipped when the drafter says "answerable" but no claim passed step 4 | Failed call, timeout or incomplete judgments follow `on_verifier_error`: `return_unverified` (return the draft, status `unavailable`) or `escalate` |
| 6. Decide | Truth table in `grounding.decide`: all claims supported is `grounded`; a mix is `partly_grounded`; none, or a settled "the passages do not answer this", is `ungrounded`. A draft with claims that is short of `grounded` is escalated once; a draft that abstains is escalated only when the verifier positively says the passages do hold the answer. On escalation `escalation_tier` gets the question and a capped set of passages, never the rejected draft: the passages the first draft cited, then the best-ranked of the rest, deduplicated, up to `escalation_token_budget` estimated tokens (3000 as shipped; null or 0 sends everything the drafter saw). An abstaining draft cited nothing, so it is the best-ranked passages alone. Its output passes the schema gate and the quote check only, against the passages it was sent | If the escalation fails, the first draft is returned with its status and `repair_error_type`. The escalation never takes the `on_error` hop. A refusal (`finish_reason: content_filter`) is its own final outcome, `refused`: no retry, no other tier, no rewording |

Abstention. `abstained` is true and the answer is one fixed sentence when the drafter
says the passages do not hold the answer and the verifier agrees, is uncertain, or is
unavailable (no escalation in any of the three), when
the escalated answer says so, or when no claim of the escalated answer passes the quote
check.

Escalation input and its trade-off. Sending everything the drafter saw cost about 0.19 USD
per attempt on the large corpus (21k to 24k prompt tokens). With the 3000-token cap the
same two cases come to 13 and 18 passages, an estimated 5.7k and 6.4k prompt tokens, about
0.05 USD of input each (computed offline from the cached passages, not yet run keyed). The
price: a passage that ranked low and that the draft did not cite is invisible to the
escalation too, so a draft that failed because retrieval ranked the needed passage low
cannot be rescued by it. `escalation_passages` and `escalation_tokens_estimated` record
what each escalation was sent.

Refusals. A reply with `finish_reason: content_filter` is recorded as error type
`refused` whatever partial text came with it. In openbook mode it is final wherever it
happens: a refused draft fails the run without the `on_error` hop, and a refused
escalation leaves the checked first draft standing. Nothing is reworded or retried.

Two grounding verifier tiers ship in `config/models.yaml`:

- `ground_jev` (default) - TypeSafe Jev through OpenRouter's Decisions API. One request
  carries one yes/no judgment per claim ("does the cited passage state or directly entail
  it") and one on whether the passages answer the question. Probabilities at or above 0.7
  read as yes, at or below 0.3 as no, and the band between as uncertain; an uncertain
  claim is not a supported claim.
- `ground_chat` - a local chat model (`qwen3-14b-ctx40k`) returning the same judgments as
  a `GroundingReport` under schema validation. A report that does not judge every claim
  exactly once is a verifier error.

Question wording, thresholds and the truth table are together in
`src/inference_arbiter/grounding.py`. A model never judges its own draft: when the drafting tier
runs the same model as the verifier, `alt_grounding_verifier_tier` is used, and without a
usable one the draft is returned as `unavailable`.

Result fields (`RunResult`; the starred ones are also columns on `runs`): `citations`
(`source`, `passage_id`, `quote`), `grounding_status`* (`grounded`, `partly_grounded`,
`ungrounded`, `unavailable`, `skipped`), `abstained`*, `passages_retrieved`*,
`sources_count`*, `retrieval_method`*, `retrieval_note`, `claims_total`*,
`claims_supported`*, `grounding_verifier_tier`*, `grounding_notes`*,
`escalation_passages`*, `escalation_tokens_estimated`* (both null when no escalation was
attempted; the estimate is passage characters / 4). `repaired` means
"escalated" and `repair_error_type` is set when the escalation failed. Reading them
together:

- `grounding_status` is the verdict on the first draft. After an escalation the returned
  answer has passed the quote check but no verifier has seen it.
- `citations` are the claims of the returned answer that passed every check that ran, and
  `claims_supported` counts them.
- `status: ok` with `partly_grounded` or `ungrounded`, `repaired: false` and
  `abstained: false` means the escalation failed and the answer is a draft that was not
  fully grounded. Without `OPENROUTER_API_KEY` every such draft ends this way.

Stuffed mode is what auto picks when the sources fit, and the eval baseline when forced
with `--stuffed`. Auto answers with `auto.closed_book_tier` or `--tier`; with `--stuffed`
triage or `--tier` picks the tier. One
call gets the full extracted text of every source ahead of the question. No retrieval, no
citations, no verification, no fallback hop. It needs the `openbook:` block for the
source-loading limits only. When the tier sets `context_window_tokens` and the prompt is
estimated above it, the run fails with `context_overflow` and no model is called.

Measured live on 2026-11-21 (two spec documents, 30 passages, about 5,100 estimated
tokens, drafter `mid`, verifier `ground_chat`, no OpenRouter key):

- With retrieval forced (`top_k` 6, hybrid): four of five questions ended correctly with
  verbatim quotes, including one unanswerable question (abstained) and two that each
  needed two passages. The fifth, a two-part question, abstained wrongly: retrieval ranked
  the passage for its second half ninth.
- Retrieval alone on 15 questions (17 needed passages, top 6): BM25 17, hybrid 16,
  embeddings 15. The questions were written from the documents' own wording, which
  favours BM25.
- The stuffed baseline answered that fifth question and the unanswerable one correctly in
  about 3 s each, against 6 to 28 s for the pipeline. For that reason
  `full_context_token_budget` ships at 6000: a corpus this small skips retrieval, and with
  every passage in the prompt the pipeline answered the fifth question with two correct
  citations.

Since 2026-11-22 the shipped config sets `retrieval_token_budget` and
`full_context_token_budget` to 12000: the drafter reads every passage of a corpus that
fits 12,000 estimated tokens, and the best-ranked 12,000 tokens' worth of a larger one.

## Execute mode

```bash
uv run python -m inference_arbiter ask "What is the sum of the decimal digits of 7 raised to the power 500?" --execute
```

Execute mode is opt-in and needs the Docker Sandboxes CLI (`sbx`, built against v0.54.0).
Plain control flow with a fixed ceiling of two model calls and two sandbox runs (four and
four with the cross-check, below). There is
no triage call: the tier is `execute.tier` (`mid`) unless `--tier` forces one.

| Step | What happens | On failure |
| --- | --- | --- |
| 1. Draft | The tier returns an `ExecuteDraft` under schema validation: `working`, `program` (Python 3, standard library only, prints its explanation and then the answer as its last line) and `claimed_answer` (what it expects that line to be, or `null`: the prompt tells the model to leave it null rather than work a calculation out by hand, see "Why the claim is optional") | A failed call or invalid output fails the run (`draft_schema_invalid` or the call's error type). No fallback hop: there is no claim to return |
| 2. Run | The program goes through the dispatch gate (below) into a fresh sandbox | A typed error: `timeout`, `nonzero_exit`, `output_too_large` (the program's fault) or `sandbox_unavailable`, `network_policy_not_applied`, `runner_not_allowed`, `program_too_large` (not the program's fault) |
| 3. Compare | Only when there is a claim. In code, no model. The executed answer is the last non-empty line of stdout. Both sides are normalised (NFKC, an `Answer:` style label dropped, surrounding quotes, backticks and asterisks stripped, whitespace collapsed, case folded, one trailing full stop dropped). They agree when the strings are equal, or when each holds exactly one number, at least one side is nothing but that number, and the numbers differ by at most 1e-9 relative. So `24 legs` agrees with `24`, and `3.14` does not agree with `3.14159` | - |
| 4. Decide | The truth table in `execute.decide` | - |
| 4a. Explain | No model call. Whatever the returned program printed before its answer line is the explanation (`executed_explanation`, capped at 4,000 characters) and goes into the answer as printed | Nothing printed: the answer has no explanation section |
| 5. Repair | Only for a program that failed, timed out or printed nothing: the same tier gets the question, the program and the tail of its stderr, once, and the new program runs once in a new sandbox. The repair call never takes the `on_error` hop | The first claim is returned unverified and `repair_error_type` is set. With no first claim the run fails (`execute_no_answer`) |

The decision table (`execute.decide`, tested for all 280 combinations of first run, repair, repairs allowed and which claims exist):

| First run | Repair | Result |
| --- | --- | --- |
| ran cleanly, output agrees with the claim | - | `claim_matched`: the executed output |
| ran cleanly, output disagrees | - | `corrected`: the executed output wins |
| ran cleanly, the model gave no claim | - | `executed`: the executed output, cross-checked by nothing |
| sandbox or gate failed | - | `unverified`: the first claim. No repair, the program was not at fault |
| program failed or timed out | agrees | `claim_matched`: the repair's output, `repaired: true` |
| program failed or timed out | disagrees | `corrected`: the repair's output, `repaired: true` |
| program failed or timed out | ran cleanly, no claim | `executed`: the repair's output, `repaired: true` |
| program failed or timed out | failed too | `unverified`: the repair's claim, or the first claim when the repair gave none |
| program failed or timed out | repair call failed or invalid | `unverified`: the first claim |
| program failed or timed out | `max_repairs: 0` | `unverified`: the first claim |

An `unverified` result returns a claim, so it needs one. Where a row says `unverified` and
the model gave no claim, the result is `failed` instead: the run has status `error`, error
type `execute_no_answer` and no answer. The sandbox runs are still in the audit table.

The answer text says which of these happened, with the answer on the last line:
`Matches the model's claim: ...`, `Corrected by execution: the model claimed "1912", but ...`,
`Computed by execution, not cross-checked: ...`
or `UNVERIFIED: this answer was not confirmed by execution (...)`, then `Answer: <value>`.
Result fields (`RunResult`, starred ones also on `runs`): `execution_status`*
(`claim_matched`, `corrected`, `executed`, `unverified`, `failed`, `skipped`, and with the
cross-check `verified`, `disputed`, `one_sided`), `claimed_answer`*
(null when the model declined to claim), `executed_answer`*,
`execution_note`, `executed_explanation`, `sandbox_runs` (one entry per program run),
`repaired`*. `executed_explanation` has no column of its own: it is a slice of
`sandbox_runs.stdout`, which is stored in full.

The answer text has three parts: the note, then `Explanation printed by the program:`
with those lines, then `Answer: <value>` as the last line. A program that prints its own
`Answer:` label keeps it in `executed_answer`, and the answer line still carries one label.

Where the explanation comes from. The prompt tells the model to make the program print
its reasoning before the answer: what was checked, the intermediate values that matter
and why the result follows, each built from values the program computed. That text is
shown as printed. Two alternatives were weighed and not taken. An explanation field in
the draft, written without values, cannot say which branch a check took or by how much a
total was off, which is what an explanation-type question asks for. A second model call
that writes the answer from the question and the stdout adds a call and lets a model
restate numbers, so the executed output would no longer be the only source of them. A
printed explanation is as wrong as its program: it explains what the program did.

Thinking level (`execute.think`, Ollama tiers only, shipped as `medium`). Printing an
explanation makes the program longer, and at the tier's default the mid model drafts the
whole program in its hidden reasoning and then writes it again: draft calls roughly
doubled. `execute.think` sets the `think` value for the draft and repair calls only; no
other mode is affected. Measured on all 26 cases of `eval/hard_questions.yaml`
(`execute:mid`, free tier, one run per row, 2026-11-23):

| `execute.think` | Pass | Executable | Reasoning | Domain | Trap | p50 / p95 | Draft call mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unset (tier default) | 26 of 26 | 15/15 | 6/6 | 2/2 | 3/3 | 20.8 s / 49.1 s | 22.7 s |
| `medium`, trial | 26 of 26 | 15/15 | 6/6 | 2/2 | 3/3 | 13.0 s / 25.5 s | 11.8 s |
| `medium`, shipped run (`hard_execute_mid_run3.json`) | 25 of 26 | 14/15 | 6/6 | 2/2 | 3/3 | 17.2 s / 40.9 s | 15.2 s |
| `low` | 24 of 26 | 14/15 | 6/6 | 2/2 | 2/3 | 9.0 s / 16.0 s | 6.3 s |

Every answer in all four runs had status `executed`, with no errors. The two `medium`
rows are the same configuration run twice: the spread between them (4 s of p50, one
case) is run-to-run variation of the free cloud model, so a one-case difference between
any two rows is not a finding. HX-12 is the executable case that failed, in two of the four runs and with a different
wrong program each time, which nothing cross-checks. Before the explanation was printed,
the three trap cases failed the rubric judge 0 of 3 for a missing explanation; they pass
3 of 3 at default and `medium`. `low` is the fastest and gave the thinnest explanations
(its HT-05 answer did not name the tag that overran).

What "claim_matched" means. The program ran to completion in the sandbox and printed what
the model claimed. It does not mean the program is a correct model of the question: a
program that prints a constant matches a constant, and a model that misreads the question
can write a wrong program and predict its output correctly. Both sides come from one
model call. This status was called `verified` until telemetry schema 8; it was renamed so
that `verified` has one meaning, the cross-check's (below). Old rows are rewritten by the
migration.

What "executed" means. The program ran to completion and the model gave no answer of its
own, so there was nothing to compare. It is weaker than `claim_matched` by exactly that
comparison, and it is the expected status for a real calculation. `claim_matched` and
`corrected` both need a claim that code compared.

Why the claim is optional. With a required claim, the mid model worked the answer out by
hand in its hidden reasoning before writing anything. Measured on one hard case
(a CRC-16 over 14 bytes, streamed on 2026-11-22): 14,341 characters of reasoning, bit by
bit, against 684 characters of JSON, 50 s; the same case had hit the 120 s tier timeout in
the eval run before it, as had two others of the 15 executable hard cases, so no program
ever ran for them. With the claim optional and the prompt saying not to compute by hand,
the same three draft calls took 4 to 8 s and returned `claimed_answer: null`.

Before and after on the 15 executable hard cases (`eval/hard_questions.yaml --category
executable`, `execute:mid`, free tier, one run each, 2026-11-22):

| | Claim required | Claim optional |
| --- | --- | --- |
| Graded pass | 12 of 12 | 15 of 15 |
| Errors (draft call timed out at 120 s) | 3 (HX-04, HX-07, HX-10) | 0 |
| Status | 8 `claim_matched` (then named `verified`), 4 corrected | 15 executed |
| Latency p50 / p95 | 42.7 s / 87.5 s | 11.4 s / 20.2 s |
| Whole run | 892 s | 173 s |

The model declined the claim on all 15, so on real calculations `claim_matched` is now rare and
`executed` is the normal result. The four `corrected` runs of the first column show what
the claim was worth there: wrong on 4 of 12. One run each, so the pass counts are within
noise; the latency and timeout difference is not.

### Cross-check: two programs (`--cross-check`, off by default)

```bash
uv run python -m inference_arbiter ask "What is the sum of the decimal digits of 7 raised to the power 500?" --cross-check
```

A single executed program can run cleanly and be wrong, and nothing notices (HX-12 in
the table above). With the cross-check a second program, B, is written and run, and
code compares the two outputs. `verified` now has one meaning: two independently written
programs printed matching values.

Four rules, each enforced in code and tested:

1. Independence. Program B is generated from the original question only. One function,
   `execute.run_program_track`, builds every message for one program, and it has no input
   that could carry the other program: not its code, output, explanation or error. That
   holds for B's repair too, which sees only B's own program and B's own failure. The test
   `test_nothing_of_program_a_reaches_program_b_and_nothing_of_b_reaches_a` plants marker
   text in everything A produces and fails if any of it is in a message sent for B (and
   the reverse), over eight scenarios including both repairs; a second test proves the
   check fires on a deliberately leaky prompt builder.
2. Diversity. B is written by `execute.cross_check_tier`. A different free model for B is
   the stricter test, but the ones available measured too weak (table below), so the
   shipped value is `mid`, the same tier as A. When both tiers run the same model, B's
   prompt asks for a different method; it still says nothing about A.
3. One form for the answer line (`execute.BARE_ANSWER_LINE_RULE`). In a cross-check both
   programs' draft and repair prompts end with the same fixed rule: the last line holds
   only the bare value the question asks for, and every reason or qualification goes in
   the explanation lines above it. A stated answer format is followed exactly; a number
   is printed alone; a question that asks to confirm something gets exactly `Yes` or
   `No`; a value that does not exist or must not be given (invalid input, a false
   premise) gets exactly `None`. The rule is a constant, the same text for A and B, so
   it carries nothing from one program to the other. Single-program execute mode keeps
   its prompts. The comparison below is not loosened to make wording agree.
4. Fail-closed comparison (`execute.values_match`, no model). Both answer lines are
   normalised as in step 3 above, then the first rule that applies decides:

| Both sides are | Match when |
| --- | --- |
| integers (decimal digits or `0x` hex) | equal, exactly |
| numbers, at least one with a decimal point or exponent | within 1e-9 relative; when both have decimal places and one has fewer, within half a unit of its last place (1580.17 matches 1580.1687, not 1580.18; 1580 does not match 1580.17) |
| lists of two or more numbers | same length, each pair matches by the rules above |
| anything else | the normalised strings are equal |

A leading currency sign and thousands separators are ignored in a number. A percent sign,
a unit or any other text makes a side a string, so `24 legs` does not match `24`, and two
sentences that mean the same in different words do not match. That last point is the
cost of deciding in code: see the measurements.

Each program takes the same course (`execute.decide_track`, all 30 combinations tested):
draft, run in its own fresh sandbox, and at most `max_repairs` repairs of its own. Then
the verdict (`execute.decide_cross_check`, all 48 combinations tested):

| Program A | Program B | Outputs | Result |
| --- | --- | --- | --- |
| ran | ran | match | `verified`: A's output, with A's printed explanation |
| ran | ran | differ, or not compared | `disputed`: both outputs and both explanations are shown; there is no `Answer:` line and the last line says no answer is confirmed |
| ran | did not | - | `one_sided`: A's output, marked `NOT CROSS-CHECKED`, with why B produced nothing |
| did not | ran | - | `one_sided`: B's output, marked the same way |
| did not | did not | - | `failed`: no answer, the run is an error (`execute_no_answer`) |

"Did not" covers a failed draft call, an invalid draft, a program that failed after its
repair, and a sandbox that could not run it. A model's claimed answer plays no part in a
cross-check; it is still stored in the audit row.

Choices made here, and why:

- A disagreement returns no answer. Picking A because its tier is stronger would turn a
  detected disagreement back into an unchecked answer. The text gives both values and
  who wrote each, so the reader can decide.
- One side failing returns the other side's output with an answer line. It is exactly
  as trustworthy as a plain `executed` answer, and the note says so. It has its own
  status because a cross-check was asked for and did not happen.
- Each program gets its own repair. A repair is fed only its own program's failure, so
  independence holds, and without it a syntax slip on the weaker tier would cost the
  whole check.
- The claim-based status was renamed to `claim_matched` (telemetry schema 8 rewrites old
  rows), so `verified` never means "the model predicted its own program's output".
- `verified` shows A's explanation; B's is in `sandbox_runs.stdout`. `disputed` shows
  both, each capped at 2,000 characters, since the reader has to weigh them.
- The two programs are drafted and run at the same time (`cross_check_parallel`), so the
  added latency is the slower of the two, not the sum.

What `verified` does not mean: both programs can misread the question the same way. Two
models make that less likely than one; it is not proof.

Telemetry: every program run is a `sandbox_runs` row with its `program_label` (`a`,
`b`), full program, stdout, stderr and outcome, whatever the verdict. `runs` has
`cross_check`, `cross_check_tier`, `executed_answer` (A), `executed_answer_b` and
`cross_check_rule` (which comparison rule decided). Program B's model calls have roles
`draft_b` and `repair_b`.

Eval: `--strategies crosscheck:<tier A>+<tier B>` (or `crosscheck:<tier>` for one tier).
A disputed run is scored as a fail, since no answer was returned. The scorecard has a
column per status and, for disputed runs, a table of which program held the expected
value (each program's output graded on its own; numeric, exact and contains cases).

Measured on all 26 cases of `eval/hard_questions.yaml`, free tiers only, 0 USD, one run
per row, 2026-11-23. Program A is always `mid` (kimi-k3.1-pro, cloud) at `think: medium`.
"Right" is the grader's pass on the returned answer; a disputed run returns none.

| Program B | `verified` (right) | `disputed` | `one_sided` (right) | `failed` | Answered right | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| none: `execute:mid`, run 3 | - | - | - | - | 25 of 26, all `executed` | 17.2 s / 40.9 s |
| `code`, qwen3 14b, local (`hard_execute_crosscheck_run1.json`) | 6 (6) | 18 | 2 (2) | 0 | 8 of 26 | 23.3 s / 46.7 s |
| `code` pointed at qwen2.5-coder 32b, local (`..._run1_mid_coder32b.json`) | 9 (9) | 12 | 5 (5) | 0 | 14 of 26 | 34.5 s / 83.9 s |
| `mid` again, asked for a different method (`..._run1_mid_mid.json`) | 22 (22) | 3 | 1 (1) | 0 | 23 of 26 | 27.0 s / 49.9 s |
| the same, with the answer-line rule in both prompts (`..._run2_mid_mid.json`) | 24 (24) | 1 | 1 (1) | 0 | 25 of 26 | 32.2 s / 47.5 s |

What the runs show:

- No `verified` answer was wrong: 61 of 61 across the four runs. The three trap cases
  are rubric-graded by the `mid` judge, the model that wrote the programs.
- HX-12, the case that ran cleanly and was wrong in run 3, was caught both times program
  A got it wrong again (7466 in the `mid`+`mid` and the qwen2.5-coder runs): B printed
  something else, so the result was `disputed` instead of a wrong answer. In the qwen3
  run A was right on HX-12 and B was wrong, also `disputed`.
- A weak program B costs most of the answers. On the cases where each program's own
  output can be graded without a judge (23 of 26), B alone was right on 6 of 21 outputs
  with qwen3, 9 of 18 with qwen2.5-coder and 23 of 23 with `mid`. Of the 30 disputes in
  the two local-model runs, 24 could be graded per program: A was right in 23, B in none,
  and in one (HX-12) both were wrong; the other 6 are trap cases. So with a local B a
  dispute nearly always means "B is wrong", and the user loses an answer A had right.
- The two local models also failed to produce a runnable program more often (`one_sided`
  2 and 5, against 1), even with their own repair.
- String answers disputed on wording until the answer line was given one form. In run 1
  of `mid`+`mid`, 2 of the 3 disputes were trap cases where both programs reached the
  same conclusion and worded the last line differently: HT-05 `No amount posted:
  malformed message` against `Malformed message`, HT-06 `No` against `No: it returns
  [-3333, ...]`. Run 2 adds rule 3 to both prompts and changes nothing in the
  comparison: HT-05 printed `None` twice and HT-06 `No` twice, both `verified` and both
  graded right; all 24 verified answers were right. The one dispute left is HX-12
  (A 7466, wrong; B 25775, right), which is the dispute the check exists for. HR-10 was
  one-sided in both runs (one draft call returned invalid output). A program that
  ignores the rule and qualifies its last line is still disputed.
- Latency: the programs run in parallel, so the run takes as long as the slower one.
  Against run 3 that is about 6 s more at p50 with qwen3, 10 s with `mid`+`mid` and 17 s
  with qwen2.5-coder (which also has to load 28 GB).

One run per row and 26 cases, so a difference of a case or two is noise; the gap between
6, 9 and 22 verified is not. The shipped `cross_check_tier` is `mid`: a second
model is the stricter test, but `code` verifies only 6 of 26 on this set, and in the
gradable disputes of the two local-model runs A was right in 23 of 24 and B in none.
`mid`+`mid` loses 3 answers to disputes, one of them the wrong HX-12 answer. Its known
weakness is that two drafts from one model can share a misreading; this set did not show
one. Try a different model with `--cross-check-tier NAME`.

### The dispatch gate (`src/inference_arbiter/sandbox.py`)

`sandbox.run_program(program)` is the only way a program reaches a sandbox. For every
program it creates a new sandbox, checks it, runs the program and removes the sandbox:

1. `sbx create --name inference-arbiter-<12 hex> --deny-network '**' --cpus 2 --memory 1g shell`,
   with no path argument, so no workspace and no host directory is mounted.
2. `sbx policy check network --sandbox <name> --json pypi.org:443` (a host the
   global policy allows). Unless sbx answers `"allowed": false` the program is not run
   (`network_policy_not_applied`).
3. The program is written to `/sandbox-run/program.py` over the stdin of an `sbx exec`,
   into a directory owned by `nobody`.
4. `sbx exec -u nobody -w /sandbox-run <name> env -i PATH=/usr/bin:/bin HOME=/sandbox-run LANG=C.UTF-8 timeout -k 2 <timeout_s> python3 -I -B program.py`.
   A second, longer timeout runs on the host. Each of stdout and stderr is capped at
   `max_output_bytes`; going over stops the program and fails the run.
5. `sbx rm --force <name>`, in a `finally`, with one retry. `sandbox_removed` records
   whether it worked.

What is isolated, as measured on 2026-11-22 with the probe in `tests/test_sandbox_live.py`
(host process had `OPENROUTER_API_KEY` set to a made-up value):

| Claim | How it is enforced | What the probe found |
| --- | --- | --- |
| No host credential | Every `sbx` subprocess gets `PATH` and `HOME` only, so no host variable can be forwarded; the program runs under `env -i` with three fixed variables | Environment was exactly `HOME`, `LANG`, `PATH`; no key, no canary value anywhere in the output |
| No host files | No workspace, no `--mount`; the program arrives over stdin | `/Users`, the host home directory, `/Volumes`, `/mnt`, `/host` do not exist; the run directory held `program.py` only |
| No network | A deny-all rule scoped to the one sandbox, confirmed with `sbx policy check` before the run. The global sbx policy is never read for writing or changed | DNS failed; TLS to `pypi.org` and `openrouter.ai` was cut; the host's Ollama on port 11434 returned no bytes; the sbx proxy answered 403 |
| Not root | Runs as `nobody` (uid 65534), not the image's `agent` user, which has passwordless `sudo` | `sudo -n` refused; `/proc/1/environ` unreadable |
| Nothing left behind | The sandbox is removed after every program | `sbx ls` before and after were identical |

What is not isolated or not guaranteed:

- The question, the program and its output go to the model tier (`mid` is a cloud model).
  Isolation is about what the program can reach, not about what the model sees.
- A `connect()` to any address succeeds inside the sandbox: sbx's transparent proxy accepts
  the TCP connection and then passes no bytes. A program that only tests whether a socket
  opens will believe it has network. No data crossed in any probe.
- The sandbox image sets its own variables for the default user (`GH_TOKEN`, proxy
  settings and others). `GH_TOKEN` there is sbx's `proxy-managed` placeholder, not a host
  token, and `env -i` keeps all of them out of the program's environment.
- The limit on memory and CPU is the size of the MicroVM (`cpus`, `memory`); there is no
  per-process limit inside it, and no limit on disk writes other than the VM's disk.
- Python only, standard library only. There is no package index to install from.
- `sbx` must be installed and its daemon running. Without it every program ends
  `sandbox_unavailable` and the answer is returned `UNVERIFIED`; nothing runs on the host.
- Removal runs in a `finally`, so an error, a timeout or Ctrl-C still removes the sandbox.
  A harness process that is killed outright (SIGKILL, power loss) leaves its
  `inference-arbiter-<hex>` sandbox behind; `sbx ls` shows it and `sbx rm <name>` removes it.
- Other sandboxes on the machine are never touched: the gate only creates and removes
  names starting with `inference-arbiter-`, and never runs `sbx policy allow/deny`, `reset`,
  `prune`, `settings`, `secret` or `daemon`.

Measured timings (2026-11-22, Mac Studio, sbx v0.54.0, `shell` kit image cached):

| Step | Time |
| --- | --- |
| `sbx create`, first time (image resolved and pulled) | 16 s |
| `sbx create`, image cached | 2.3 s |
| `sbx exec` of a trivial command | 0.15 to 0.3 s |
| `sbx rm --force` | 0.3 s |
| The whole gate for one small program (create, check, stage, run, remove) | 3.1 to 3.4 s |
| `ask --execute` end to end on `mid`, three questions | 15 to 21 s, of which the draft call 12 to 17 s |

A new MicroVM per program costs about 3 s, small next to the model call, so the gate
does not reuse a sandbox between programs.

First eval numbers (2026-11-22, the 12 numeric cases of the golden set, one run each, free
tier only, `scripts/run_eval.py --strategies tier:mid execute:mid`): `execute:mid` 12/12,
all 12 `verified`, none corrected, unverified or repaired; `tier:mid` 11/12 (it missed
RS-05). Twelve cases answered once is within noise, and it says nothing yet about
questions outside arithmetic and code tracing. Latency p50 was 8.1 s for
`execute:mid` against 2.9 s for `tier:mid`; the 12 sandbox runs averaged 3.1 s each and
all 12 sandboxes were removed.

## Codetest mode

```bash
uv run python -m inference_arbiter ask "Write a Python function to_minor_units(amount: str, exponent: int) -> int that converts a decimal amount string to minor units, rounding half to even." --code-test
```

Codetest mode is opt-in, for a question that asks for a Python function. It needs the
`sbx` CLI and the `execute:` block (the sandbox limits) as well as its own `code_test:`
block. Auto mode never picks it. Plain control flow in
`src/inference_arbiter/codetest.py`, with a fixed ceiling of four model calls and two
sandbox runs and no triage call:

| Step | What happens | On failure |
| --- | --- | --- |
| 1. Interface | `code_test.interface_tier` (`local`) gets the question and returns a typed `CodeInterface`: the function's name and its `def` line. `codetest.check_interface` checks it in code: it must parse as one plain function definition with that name, and when the question itself holds a `def name(` line the name must be that one. The module name is fixed: `solution` | The run fails (`interface_invalid`, `interface_mismatch` or the call's error). Nothing else is called |
| 2. Solution | `code_test.solution_tier` (`mid`, or `--tier`) gets the question and the interface and returns the module | The run fails: there is nothing to return |
| 3. Tests | `code_test.tests_tier` (`local`, or `--tests-tier`) gets the question and the interface and returns a `unittest` module. It is not shown the solution | The solution is returned as `untested` |
| 4. Run | One sandbox run of both, through the dispatch gate above, unchanged | A typed gate error, or a run class (below) |
| 5. Decide | The truth table in `codetest.decide` | - |
| 6. Repair | Only when the tests failed the solution or the solution could not be loaded: the solution tier gets its own module and what happened, once, and the new module runs once against the same tests. The tests are never rewritten | The first result stands; `repair_error_type` is set |

Steps 2 and 3 run at the same time by default (`code_test.parallel`); neither is an input
of the other.

**Independent tests.** The test writer's whole prompt is the system prompt, the question
and the interface (`codetest.build_tests_messages`, a function with no parameter that
could carry a solution). Nothing of the solution, of its run or of its repair is passed to
any call on the tests tier. `tests/test_codetest.py` plants a marker in every field of
the solution drafts and in the sandbox output, runs every path including both repairs,
and asserts that the marker never reaches the test writer or the interface call; a second
test swaps in a builder that leaks and shows the check fails. The shipped config writes
the solution and the tests on two different free models, so one model does not pass its
own mistake; `--tests-tier` picks another. Both writers do share one thing by design: the
interface, which comes from the question alone.

**One program for the gate.** The gate runs one program and stages nothing else, and it
is not changed for this mode. The program is a fixed driver
(`src/inference_arbiter/codetest_driver.py`, standard library only, never written by a
model) with the solution and the tests embedded as string literals. It loads the solution
as the module `solution`, loads the tests, runs them with `unittest`, and prints one
result line with the counts. The exit code alone is never trusted: a run passes only when
the exit code is 0 and the result line says every test passed
(`codetest.classify_test_run`), so a solution that ends the process early is "not
finished", not a pass. Whatever the solution and the tests print is captured inside the
driver and cannot reach the result line.

Solution, tests and driver run in **one process**. That is safe for the host (the whole
process is inside the sandbox) but it is not a barrier between the three: code that set
out to forge the result line or to patch `unittest` could do it. The mode guards against
mistakes, not against a solution written to cheat its own tests.

**Tests that check nothing.** When every test passes, the driver runs the same tests once
more against a stub: a copy of the solution module whose function raises on every call.
Tests that pass against the stub too do not depend on what the function returns, and the
result is `untested`, not `tests_passed`.

How one sandbox run is classed (`codetest.classify_test_run`, first match wins):

| # | What came back | Class |
| --- | --- | --- |
| 1 | A gate or sandbox error | `infra_failed` |
| 2 | Timeout, or output over the cap | `not_finished` |
| 3 | No well-formed result line, or an exit code that does not match it | `not_finished` |
| 4 | The solution did not load, or lacks the function | `solution_invalid` |
| 5 | The tests did not load, the run was cut short, or no test ran | `tests_invalid` |
| 6 | At least one test failed or raised | `tests_failed` |
| 7 | Every test passed, and passed against the stub too | `vacuous` |
| 8 | Every test passed, and the stub run did not | `passed` |

The decision table (`codetest.decide`, tested for all 126 combinations of first run,
repair and repairs allowed):

| # | First run | Repair | Result |
| --- | --- | --- | --- |
| 1 | passed | - | `tests_passed`, the first solution |
| 2 | vacuous, tests_invalid, not_finished, infra_failed | - | `untested`, the first solution. No repair: nothing here is the solution's proven fault |
| 3 | tests_failed or solution_invalid | not attempted, allowed | Attempt the one repair |
| 4 | tests_failed | not attempted, not allowed | `tests_failed`, the first solution |
| 5 | solution_invalid | not attempted, not allowed | `failed`: no answer, the run is an error |
| 6 | either | passed | `tests_passed`, the repaired solution |
| 7 | either | tests_failed | `tests_failed`, the repaired solution |
| 8 | either | vacuous | `untested`, the repaired solution |
| 9 | tests_failed | anything else | `tests_failed`, the first solution: it is the one known to run |
| 10 | solution_invalid | tests_invalid, not_finished, infra_failed | `untested`, the repaired solution |
| 11 | solution_invalid | solution_invalid, or the repair call failed | `failed`: no answer, the run is an error (`solution_invalid`) |

What the answer says. The answer is a note followed by the solution in one code fence:

| `execution_status` | The note begins |
| --- | --- |
| `tests_passed` | "Passed its generated tests: N of N unit tests, written by tier T, model M from the question alone, passed in an isolated sandbox. The tests were written by a model and were not checked against a reference, so this does not show the solution is correct." |
| `tests_failed` | "FAILED its generated tests: K of N unit tests ... failed in an isolated sandbox (the failing tests). ... a failure can be a wrong test as well as a wrong solution." |
| `untested` | "NOT TESTED: (why). The solution below was not run against any test that could fail it." |

`tests_passed` is never called verified. `verified` keeps its one meaning, two
independently written programs printing the same value in a cross-check. Passing tests
that a model wrote from the question are evidence that the solution does what that model
understood the question to ask, and no more: both writers can misread the question the
same way, and the tests can be too weak to see a bug. How often that happens is what the
code set measures (`eval/README.md`, "Code set"): there, a solution is graded by fixed
reference tests that no model sees, and the report counts how often the generated tests
caught and missed a solution the reference tests failed.

The run row stores `mode` `codetest` and the status in `execution_status`; the result
also carries `code_function`, `code_signature`, `code_solution`, `code_first_solution`,
`code_tests`, `tests_tier`, `tests_run`, `tests_failed` and `code_first_outcome` (how the
first solution's run was classed, kept apart from the repaired one's). Each sandbox run
is a `sandbox_runs` row whose `program` is the whole program the gate ran and whose
`executed_answer` is the result line.

Measured 2026-11-24 on the code set (20 cases, one run each,
`eval/results/2026-11-24-code-tests.md`), with the shipped `code_test:` block
(`codetest:mid+local`) against a single `mid` call:

- Both got 20 of 20 by the reference tests, and `mid` was right on every first attempt, so
  the mode added no accuracy on this set. Median latency 28.8 s against 19.2 s.
- The tests written by `local` are not reliable. The run reported `tests_failed` for 13 of
  the 20 solutions, all of which pass the reference tests; in every such case the
  generated tests also fail the case's reference solution. Treat a `tests_failed` from
  the shipped config as "the solution or the tests are wrong", as the answer text says.
- 14 repairs ran against such tests and none broke a solution.
- Only 5 generated suites accept their reference solution; they fail 8 of their 10 seeded
  wrong solutions.

Not measured: `codetest:mid+mid` (the run did not finish), so whether `mid` writes better
tests than `local` is open, and so is the mode's worth on questions a single call gets
wrong. Auto mode never picks this mode.

## MCP mode

`ask "question" --docs DIR` answers from the `.md` and `.txt` documents of one local
folder. The documents are not read by the harness: they are served by a small MCP server
(`src/inference_arbiter/docs_server.py`) and fetched through a gate
(`src/inference_arbiter/toolgate.py`) that allows two tools, checks every call and logs
every call. The answer then comes from the same grounded pipeline as open-book mode, so
each citation is checked verbatim against text the server returned.

What this mode is and is not:

- The server is a process on this host, a stdio subprocess of the harness. No sandbox is
  involved: the sandboxes of execute mode have no network and nothing can be installed in
  them, so a server there could not be reached.
- The server is confined to the folder by its own code, not by the operating system. It
  runs as the calling user and could, if its code were wrong, read anything that user can.
- "Scoped" means: two read-only tools, and only the files indexed from one folder when the
  server started. "Per run" means: one server process per question, started inside
  `Harness.answer` and stopped before any model is given document text.
- Auto mode never picks this mode.

### What one run does

| Step | What happens |
| --- | --- |
| 1. Start | The server is started on the folder. The gate asks which tools it offers; anything but exactly `list_documents` and `read_document` ends the run (`tool_list_mismatch`) |
| 2. List | Code calls `list_documents`: path, size, line count and SHA-256 of each document |
| 3. Plan | One call to `mcp_docs.plan_tier` gets the question and that listing, never any content, and returns a tool plan: a list of tool names with arguments. This is the only model call made while the server is alive |
| 4. Gate | Every planned call goes through the gate like any other. A refused call is recorded and not sent, and does not end the run. The table under "The tool plan" says what happens next |
| 5. Read | Code reads the rest of each chosen document page by page and checks the assembled text against the SHA-256 from the listing |
| 6. Stop | The server is stopped and the harness checks that the process is gone |
| 7. Answer | The documents go to the open-book pipeline unchanged: retrieval, cited draft, quote check, grounding verifier, at most one escalation |

Model calls: one plan call plus at most the five of the open-book pipeline. A failed
server call (a timeout, a server-side error, an unusable or oversized result) is never
worked around: the run ends with that typed error.

### The server

- Root: resolved once. `/`, the home folder and any folder above it are refused.
- Index: built once at start, without following symbolic links. A file is served only if
  it is a regular file, not a link, ends in `.md` or `.txt`, has a plain name (letters,
  digits, `.`, `_`, `-`; no leading dot, so never `.env`), resolves to a path under the
  root, is UTF-8 text and is inside the size limits. What was skipped is counted by reason
  in the listing.
- Snapshot: the text of every indexed file is read at start and kept in memory. The
  server opens no file after that, so a file swapped for a link later changes nothing.
- `read_document` takes a path only as an exact key of the index. `..`, an absolute path
  or a link is not a key.
- No tool writes. Files are opened read-only.
- Environment: `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM` and `USER`, which the `mcp`
  client passes to any stdio server; nothing else, so no key from `.env` reaches it. The
  names are stored with the run.
- It ends when its stdin closes, and on its own after `mcp_docs.max_lifetime_s` in any case.
- Its stderr goes to a capped buffer, not to the harness's output.
- The harness registers no sampling, elicitation or roots handler.

### The gate

A call is refused, without contacting the server, by the first row that matches
(`toolgate.decide_call`, a pure function):

| # | Condition | Refusal |
| --- | --- | --- |
| 1 | The tool is not on the allow-list | `tool_not_allowed` |
| 2 | The arguments as JSON are over `max_argument_bytes` | `tool_arguments_too_large` |
| 3 | The arguments are not an object or do not fit the tool's shape | `tool_arguments_invalid` |
| 4 | A path that is absolute or contains `..`, `.`, a backslash or NUL | `path_not_allowed` |
| 5 | A path that is not one of the listed documents | `unknown_document` |
| 6 | `max_calls` calls were already sent in this run | `tool_call_limit` |

An allowed call is sent with a timeout. Its reply fails (`toolgate.classify_reply`) when
the server marks it as an error (`tool_error`), when it holds anything but text or does
not parse into the expected shape (`tool_result_invalid`), when it is over
`max_result_bytes` (`tool_result_too_large`; a result is never cut to fit) or when the run
would pass `max_total_result_bytes` (`tool_result_budget`). A server that does not answer
is `mcp_timeout`; one that cannot be started or dies is `mcp_unavailable`.

### The tool plan

The planning model's output is the only model-written input to the gate. What becomes of
it (`mcpdocs.decide_plan`, a pure function; first match wins):

| # | Plan call | Planned calls | Allowed by the gate | `plan_status` | What happens |
| --- | --- | --- | --- | --- | --- |
| 1 | failed | - | - | `call_failed` | Every listed document is read |
| 2 | output fails the schema | - | - | `invalid` | Every listed document is read. No second attempt |
| 3 | ok | none | - | `empty` | Every listed document is read |
| 4 | ok | one or more | none | `all_refused` | The run ends with `plan_refused`. Nothing is read |
| 5 | ok | one or more | some, but no read | `no_read` | Every listed document is read |
| 6 | ok | more than `max_planned_calls` | at least one read | `over_cap` | The calls past the cap are refused (`plan_call_limit`); the allowed ones run |
| 7 | ok | within the cap | some, at least one read | `partial` | The allowed calls run; the refused ones are only recorded |
| 8 | ok | within the cap | all | `ok` | The planned calls run |

Reading every listed document is what the mode does without a usable plan: the folder is
the scope the caller chose, so that grants nothing the caller did not. A plan made only of
calls the gate refuses is different: it is not replaced by a wider read.

Document names are text that whoever fills the folder controls, and the planning model
reads them. They are held to plain names of at most `max_name_chars` characters and at
most `max_documents` entries by the server and again by the gate, and the prompt tells
the model they are data. The plan can only select which listed documents are read; it
cannot name a tool or a path the gate does not already allow.

### What comes back

The answer, its citations (each with the document's SHA-256 and the `tool_calls` rows its
text came from), `plan_status`, `documents_listed`, `documents_read`,
`tool_server_stopped` and one `tool_calls` entry per call the gate saw. The trace prints
one line per tool call with who asked for it (`code` or `plan`).

Not measured: this mode has not been run with a live model. See "Known limits".

## Phase 2 status

The build plan's exit test is scorecard data proving the multi-model chain beats, or
matches at significantly lower cost, a single direct mid-tier call on the 30 to 50
question suite. Final runs on 2026-11-22 with the shipped defaults (Jev verifiers,
`frontier` escalation), 0.96 USD in all:

| Suite | Multi-model | Single mid-tier call | Exit test |
| --- | --- | --- | --- |
| Closed-book, 36 cases, 3 runs | chain 102/108, 0.4833 USD, p50 about 10 s | `tier:mid` 106/108, 0 USD, p50 about 3 s | Not met: chain won 0 cases, lost 4 |
| Open-book, small set, 26 cases | openbook 26/26, 0.0872 USD, p50 8.2 s | `stuffed:mid` 26/26, 0 USD, p50 2.6 s | Not met: a tie at higher cost |
| Open-book, large corpus, 16 cases | openbook 16/16, 0.39 USD, p50 15 s | `stuffed:mid` cannot run (sources over three times its context window); `closed:mid` 0/13 | Ahead only because the baseline cannot run |

Phase 2 as written is not passed: on the closed-book suite the exit test names, a single
mid-tier call is more accurate and free. The multi-model path is ahead only where one call
cannot hold the sources, and there the scorecard reports "right, baseline errored", never
"won".

Phase 2 is closed on this result. The original hypothesis did not hold: a verification
chain over a strong mid tier does not beat one call, so `auto` mode makes one call by
default and the chain stays behind a flag. Multi-model work pays in two places the
measurements did confirm: sources too large for one call (the open-book pipeline), and
answers that need computation (sandboxed execution with a two-program cross-check, in
`eval/results/2026-11-23-hard-set.md`).

What the runs also showed:

- The open-book pipeline's citations hold up: every citation verbatim in its source on
  both sets, and no unanswerable question answered (8/8 abstained).
- Frontier escalation is the cost and the weak point. In the open-book runs three
  escalations were attempted and two failed (truncated JSON, no content); both were billed
  (0.275 of 0.477 USD) and the mid draft was kept, correctly as it turned out. Jev
  grounding itself cost under 0.01 USD per set.
- In the chain, 13 or 14 of 36 cases escalate per run and no escalation turned a baseline
  miss into a right answer.

Scorecards: `eval/results/2026-11-21-phase2-baseline.md` (closed-book, version 1),
`eval/results/2026-11-22-phase2-openbook.md` (open-book, first keyed run),
`eval/results/2026-11-22-phase2-free-runs.md` (three keyless runs per case) and
`eval/results/2026-11-22-phase2-final.md` (final keyed runs, verdict and caveats). The open-book set, its grading
rules and how to run it (`scripts/run_eval.py --suite openbook`) are in
`eval/openbook/README.md`.

## Telemetry

`runs` has one row per question and `calls` one row per model call, with latency, tokens
and cost. Call roles are `triage`, `answer` (route and stuffed mode), `draft`, `verify`
and `repair` (chain mode), `embed`, `draft`, `ground` and `escalate` (openbook mode),
`draft` and `repair` (execute mode; `draft_b` and `repair_b` are program B's calls in a
cross-check), `interface`, `draft`, `tests` and `repair` (codetest mode), `plan` and the
openbook roles (mcp mode). `tool_calls` is the audit log of mcp mode, one row per tool call
the gate saw: `trace_id`, `seq`, `origin` (`code`, or `plan` for a call the planning model
wrote), `tool`, the `arguments` as JSON in full (`arguments_truncated` marks the one case
they are cut: a call refused for arguments over the cap), `status` (`ok`, `refused`,
`error`), `error_type`, `result_bytes`, `result_sha256`, `is_error` and `duration_ms`. A
result is stored as its size and hash only, never as text. `sandbox_runs` is the audit log of execute and codetest mode, one
row per program run: `trace_id`, `seq`, `attempt` (`first`, `repair`), `sandbox_name`,
`runner`, the full `program` text, `claimed_answer`, `executed_answer`, `status`,
`error_type`, `exit_code`, capped `stdout` and `stderr` with their truncation flags,
`duration_ms` (the program), `total_ms` (create to remove), `timed_out`, `network_policy`,
`environment_names` (names only; the values are fixed in code), `sandbox_removed` and
`outcome` (`agree`, `disagree`, `no_claim`, `program_failed`, `infra_failed`; in a cross-check
`ran`, `program_failed`, `infra_failed`; in codetest mode `passed`, `vacuous`, `tests_failed`,
`solution_invalid`, `tests_invalid`, `not_finished`, `infra_failed`, with the run's result line in
`executed_answer`) and `program_label` (`a` or `b`, which program of a
cross-check). Codetest mode added no column: its statuses (`tests_passed`, `tests_failed`,
`untested`) are values of `execution_status`. The
schema is at version 9 (version 9 adds mcp mode: the `tool_calls` table and, on `runs`,
`plan_status`, `tool_server_stopped` and `tool_server_environment`, the names of the
variables the documents server was given; version 8 adds the cross-check: `cross_check`, `cross_check_tier`,
`executed_answer_b` and `cross_check_rule` on `runs`, `program_label` on `sandbox_runs`, and it
renames the claim-based `execution_status` value `verified` to `claim_matched` in existing rows;
version 7 adds `sandbox_runs` and, on `runs`, `execution_status`,
`claimed_answer` and `executed_answer`; version 4 adds each call's finish reason and reasoning tokens, and
grounding notes on the run; version 5 adds what an escalation was sent:
`escalation_passages` and `escalation_tokens_estimated`; version 6 adds `auto_reason`, why
auto mode picked the run's `mode`, null when the caller picked it; when the computation check ran, its
tier, verdict and confidence are part of this text and its call is a `calls` row with role `check`, so
the check needed no new column); an older database is migrated in place the first time the
harness opens it (columns are added with defaults, old rows stay readable).

## Configuration

`config/models.yaml` defines providers, tiers, thresholds and prices; the field reference
is in the comments at the top of that file. Swapping a model is a YAML edit only. API keys
are referenced by environment variable name and never stored in config, logs, SQLite or
output. The sha256 of the config file is stored with every run.

## Layout

| Path | Role |
| --- | --- |
| `src/inference_arbiter/config.py` | YAML loading, pydantic validation, `.env` loader, config hash |
| `src/inference_arbiter/schemas.py` | `AskRequest`, `TriageDecision`, `ComputationCheckDecision`, `DraftAnswer`, `VerifierVerdict`, `OpenbookDraft`, `ExecuteDraft`, `CodeInterface`, `SolutionDraft`, `UnitTestsDraft`, `Citation`, `CallResult`, `RunResult` |
| `src/inference_arbiter/clients.py` | `OllamaNativeClient` (native `/api/chat` and `/api/embed`), `OpenRouterClient` (`openai` SDK), `OpenRouterDecisionsClient` |
| `src/inference_arbiter/triage.py` | Text-only triage call and output validation |
| `src/inference_arbiter/auto.py` | Auto mode's decision table: which strategy answers when the caller did not pick one |
| `src/inference_arbiter/computation_check.py` | Auto mode's computation check: the prompt, the typed call and the verdict table |
| `src/inference_arbiter/router.py` | Routing rules, auto-mode resolution, single fallback hop, telemetry hand-off; `Harness.answer(request)` is the entry point for every mode |
| `src/inference_arbiter/chain.py` | Chain control flow, schema gate, draft and repair prompts |
| `src/inference_arbiter/verifier.py` | Chat verifier prompt, Jev questions, thresholds and truth table |
| `src/inference_arbiter/sources.py` | Loading files and URLs, HTML to text, chunking into passages |
| `src/inference_arbiter/retrieval.py` | Embedding, BM25 and fused passage ranking, embedding cache |
| `src/inference_arbiter/grounding.py` | Quote check, Jev questions and chat prompt for grounding, thresholds, truth table |
| `src/inference_arbiter/openbook.py` | Openbook control flow, draft prompt, stuffed baseline |
| `src/inference_arbiter/sandbox.py` | Host-side dispatch gate: runs one Python program in a throwaway `sbx` MicroVM and removes it |
| `src/inference_arbiter/execute.py` | Execute control flow, prompts, the claim-versus-output comparison and its truth table |
| `src/inference_arbiter/codetest.py` | Codetest control flow, prompts, the interface check, reading a test run and its truth table |
| `src/inference_arbiter/codetest_driver.py` | The fixed program that runs a solution against its tests inside the sandbox and prints one result line |
| `src/inference_arbiter/docs_server.py` | The read-only MCP documents server of mcp mode: folder index, path rules, the two tools |
| `src/inference_arbiter/toolgate.py` | Host-side tool gate: the allow-list, the per-call and per-reply decision tables, one record per call |
| `src/inference_arbiter/mcp_session.py` | The stdio session to the documents server; the only module that imports `mcp` |
| `src/inference_arbiter/mcpdocs.py` | Mcp mode control flow: listing, the tool plan and its truth table, reading, hand-over to the open-book pipeline |
| `src/inference_arbiter/telemetry.py` | SQLite writer and schema migration |
| `src/inference_arbiter/cli.py` | `ask` command |
| `eval/` | Golden questions for Phase 2, the hard set, the code set with its reference tests (`eval/code_questions.yaml`, `eval/code_reference/`), the open-book set (`eval/openbook/`) and recorded scorecards (`eval/results/`) |
| `scripts/run_eval.py` | Eval runner and scorecard for both suites; grades the code set with reference tests in the sandbox |
| `scripts/check_computation.py` | Scores the computation check alone against the `computable` labels of the question files |
| `scripts/fetch_corpus.py` | Fetches and verifies the open-book corpus against `eval/openbook/corpus.yaml` |

## Known limits

- Mcp mode has not been run with a live model and has no scorecard. Everything about it
  is covered by tests with fake model clients; the server, the gate and the stdio session
  are also tested against the real server process. Not yet seen live: what plans the
  `mid` tier writes, how often the gate refuses them, and what the mode adds in latency.
  The plan schema holds a free-form `arguments` object, which a provider that enforces
  strict JSON schemas may reject; the shipped plan tier (`mid`, through Ollama) is not
  such a provider, but that too is unverified. The server is a host process confined by
  its own path checks, not by the operating system. Only `.md` and `.txt` files are
  served, HTML is not converted, and a document with a line longer than
  `mcp_docs.max_read_bytes` is skipped. Every chosen document is read in full on every
  run, so a large corpus means many tool calls; the large open-book corpus is out of reach.
- Codetest mode has no scorecard yet. Its tests are written by a model: `tests_passed`
  means the solution passed them, not that it is right. Solution, tests and the driver
  share one process inside the sandbox, so the result line is protected against mistakes
  (stray output, an early exit, a crash) and not against a solution written to forge it.
  A test run that does not finish is reported as not tested, without saying whether the
  solution or the tests ran too long. Python only, standard library only, one function
  per question.
- Openbook mode has been run live with the `mid` drafter, local embeddings and both
  grounding verifiers (`ground_chat`, and `ground_jev` on the 26-case open-book set on
  2026-11-22). On that set Jev's verdicts matched the local verifier's: 20 grounded, 6
  ungrounded, the 6 being the abstentions. The `frontier` escalation has run live five
  times (2026-11-22) and failed three times. Two were on the same unanswerable large-corpus
  case (OB-LU1): the second is a recorded refusal (`finish_reason: content_filter`, native
  `refusal`), the first had the same symptom (no content) before finish reasons were
  recorded. The third (OB-23) was JSON cut off after 80 tokens, reason not recorded. The
  draft was kept every time. A refused call is still billed for its input (0.195 USD for
  the recorded one), which the escalation token cap now bounds at about 0.05 USD. The
  capped escalation input has not been run keyed. URL fetching inside the
  harness is covered by tests with fakes only (the eval corpus is fetched to local files
  first). The 0.3 / 0.7 thresholds have not
  been tuned against the real model.
- An abstention escalates only when the verifier positively disagrees with it; agreement,
  an uncertain verdict or a verifier failure all abstain at no cost. That is right for unanswerable
  questions, and it means a retrieval miss ends in a wrong abstention, not in a frontier
  call (open-book eval case OB-21 before the 2026-11-22 retrieval change, see below).
- With the shipped config the question and the retrieved passages go to a cloud drafter,
  and with `ground_jev` to a cloud verifier as well. Fetched pages are not distilled by a
  local model first; retrieval is the only reduction before the cloud call.
- A question with two parts can lose one part at retrieval: the query is ranked as one
  string, so the passage for the weaker half can fall outside the selected passages. The
  drafter then reports the whole question as unanswerable. A fixed `top_k` of 6 lost
  open-book eval case OB-21 this way on every run; since 2026-11-22 the shipped config
  fills a 12,000-token budget instead, which reaches further down the ranking.
- A page that builds its content in JavaScript yields little text: only the HTML that
  arrives is read. PDFs and other non-text sources are refused.
- The quote check normalises whitespace and case only. A model that drops Markdown
  markers or changes typographic quotes inside a quote fails it.
- Stuffed mode checks the prompt against the context window only for tiers that set
  `context_window_tokens` (`mid` does), and the check uses the chars / 4 estimate.

- Execute mode lets the model decline the claim, and for a real calculation it usually
  does. Those answers are `executed`, not `verified`: one program, run once, checked
  against nothing. A wrong program gives a wrong `executed` answer with no warning.
- Execute mode will "verify" a question that needs no computation: the model writes
  `print("Canberra")` and claims `Canberra`. Auto mode reaches execute only through the
  computation check, which is on in the shipped config; a false yes from the check sends
  such a question there.
- A program that prints a different but equivalent form of the claim (`1/3` against
  `0.3333`, two sentences with different wording) counts as a disagreement and the
  executed output is returned as `corrected`.
- Chain mode with the shipped config sends the question and the draft to a cloud verifier
  even when the draft is local. `allow_cloud_fallback` does not govern the verifier or the
  repair tier.
- When triage routes a question to `mid`, the drafter and the `verify` verifier are the
  same model (`kimi-k3.1-pro:cloud`), so the second opinion is not independent for those
  runs.
- Chain mode has been run live with local and `mid` drafts and the `verify` verifier.
  The `frontier` repair and the `verify_jev` verifier need `OPENROUTER_API_KEY` and are
  covered by tests with fake clients only.
- Answer calls are text-only as well: non-text content parts are not forwarded to any model.
- The OpenRouter path is covered by tests with a fake SDK client only. It has not been run
  against the live API.
- Ollama cloud models have no per-token price, so their cost is logged as 0 with
  `cost_source` `local_zero` unless prices are set on the tier.
- Runs made with `--simulate` are written to telemetry like any other run. They are
  identifiable by the `simulated_*` error type on the injected call.
- The planned `mid` default `deepseek-v4-flash:cloud` was retired upstream (HTTP 410 on
  2026-11-21), so `mid` is set to `kimi-k3.1-pro:cloud`.
