"""Typed hand-offs between the CLI, triage, router and clients."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RouteLabel = Literal["local", "code", "mid", "frontier"]
TriageStatus = Literal["ok", "timeout", "error", "invalid", "skipped"]
RouteReason = Literal[
    "forced",
    "triage_timeout",
    "triage_error",
    "triage_invalid",
    "low_confidence",
    "triage_label",
    # openbook mode without a forced tier: the drafter is openbook.drafter_tier, no triage.
    "configured",
    # auto mode without a forced tier: auto.closed_book_tier answers, no triage.
    "auto",
]
Simulation = Literal[
    "triage-timeout",
    "answer-error",
    "verifier-error",
    "draft-invalid",
    "quote-mismatch",
    "grounding-error",
    "tool-refused",
]
CHAIN_ONLY_SIMULATIONS = ("verifier-error", "draft-invalid")
OPENBOOK_ONLY_SIMULATIONS = ("quote-mismatch", "grounding-error")
# The grounded pipeline also runs inside mcp mode, so its simulations apply there too.
GROUNDED_MODES = ("openbook", "mcp")
MCP_ONLY_SIMULATIONS = ("tool-refused",)
CostSource = Literal["provider_reported", "estimated", "local_zero"]
Mode = Literal["auto", "route", "chain", "openbook", "stuffed", "execute", "codetest", "mcp"]
SOURCE_MODES = ("openbook", "stuffed")
# What a finished run reports: auto is always resolved to one of the others.
EffectiveMode = Literal["route", "chain", "openbook", "stuffed", "execute", "codetest", "mcp"]
VerificationStatus = Literal["supported", "unsupported", "uncertain", "unavailable", "skipped"]
GroundingStatus = Literal["grounded", "partly_grounded", "ungrounded", "unavailable", "skipped"]
# execute mode, one program. claim_matched: the program ran cleanly and its output agrees
# with the claim of the model that wrote it (called `verified` before schema version 8).
# corrected: it ran cleanly and disagrees, the executed output is the answer.
# executed: it ran cleanly and the model gave no claim to compare, the executed output is
# the answer and nothing independent agreed with it. unverified: nothing was confirmed by
# execution, the answer is the model's claim. failed: no program ran cleanly and there is
# nothing to return, so there is no answer and the run is an error.
# execute mode with the cross-check, two programs written independently. verified: both
# ran cleanly and their outputs match (execute.values_match); this is the only meaning of
# `verified`. disputed: both ran and the outputs differ; both are shown, none as the
# answer. one_sided: one program ran and the other did not; its output is the answer,
# marked as not cross-checked. failed: neither ran.
# codetest mode, a solution and a unittest suite written by separate calls (codetest.py).
# tests_passed: every generated test passed in the sandbox, and the same tests did not pass
# against a stub that raises on every call. The tests are model-written, so this is never
# called `verified`. tests_failed: at least one generated test failed; the solution is
# returned with the failing tests named. untested: the solution was not run against any
# test that could fail it (no tests, tests that could not run, tests that check nothing, a
# run that did not finish, or a sandbox failure). failed: the solution could not be loaded,
# also after the one repair, so there is no answer and the run is an error.
ExecutionStatus = Literal[
    "verified", "disputed", "one_sided", "claim_matched", "corrected", "executed", "unverified",
    "tests_passed", "tests_failed", "untested",
    "failed", "skipped",
]
CallRole = Literal[
    "triage", "answer", "draft", "verify", "repair", "embed", "ground", "escalate", "draft_b", "repair_b",
    "check", "interface", "tests", "plan",
]
# mcp mode's tool plan (mcpdocs.decide_plan). ok: every planned call was allowed. partial:
# some were refused, the allowed ones ran. over_cap: the plan held more calls than
# mcp_docs.max_planned_calls; the calls past the cap were refused. all_refused: the gate
# refused every planned call and the run ended there. empty: the plan held no call.
# no_read: it held no allowed read. invalid: the planning model's output failed the schema.
# call_failed: the planning call itself failed. In the last four cases every listed
# document was read instead. not_run: the run ended before the plan call.
PlanStatus = Literal[
    "ok", "partial", "over_cap", "all_refused", "empty", "no_read", "invalid", "call_failed", "not_run",
]
# Auto mode's computation check (computation_check.py). not_run: the check was off, the
# request had sources or the caller picked the mode or the tier. yes: the check said a short
# program can compute the answer, at or above the confidence threshold. no: it said one
# cannot. low_confidence: it said yes below the threshold. failed: the call timed out,
# errored or returned invalid output. Only `yes` sends a question to execute mode.
ComputationVerdict = Literal["not_run", "yes", "no", "low_confidence", "failed"]

STATUS_OK = "ok"
STATUS_ERROR = "error"

PLAN_SCHEMA_MAX_CALLS = 64

ERROR_SIMULATED_TIMEOUT = "simulated_timeout"
ERROR_SIMULATED_ERROR = "simulated_error"


class Message(BaseModel):
    """OpenAI-style chat message. content is a string or a list of content parts."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1)
    tier: str | None = None
    effort: Literal["low", "medium", "high"] | None = None
    simulate: Simulation | None = None
    # auto (default): the harness picks, see auto.decide_mode. No sources: one call to
    # auto.closed_book_tier, no triage; with the computation check on, a question a short
    # program can compute goes to execute mode. Sources that fit the context window: stuffed.
    # Sources that do not fit, or require_citations: openbook.
    # route: one triage call plus one answer call. chain: draft, schema gate, verify, repair.
    # openbook: load sources, retrieve passages, draft with citations, quote check,
    # groundedness verifier, at most one escalation. stuffed: one call with the full text
    # of every source in the prompt (the baseline for openbook).
    # execute: the model writes a short Python program and the answer it claims; the
    # program runs in an isolated sandbox and code compares the two (see execute.py).
    # codetest: one call fixes the function's interface, one writes a solution, one writes
    # unittest tests from the question and the interface alone; both run in one isolated
    # sandbox and the answer says whether the solution passed its generated tests
    # (see codetest.py).
    # mcp: the documents of one local folder are fetched through a read-only MCP server
    # behind the tool gate, one model call plans which to read, and the grounded pipeline
    # of openbook mode answers from them with citations (see mcpdocs.py). Never picked by auto.
    mode: Mode = "auto"
    # chain: verify with this verifier tier instead of chain.verifier_tier.
    # openbook: ground with this verifier tier instead of openbook.grounding_verifier_tier.
    # auto with sources: used when auto picks openbook.
    verifier_tier: str | None = None
    # auto, openbook and stuffed only: local file paths or http(s) URLs, in citation order.
    sources: list[str] = Field(default_factory=list)
    # auto with sources: always take the openbook pipeline, even when the sources would fit.
    require_citations: bool = False
    # route answers only: when the answer tier's call fails, do not take the fallback hop;
    # the run fails with that tier's error. For measuring one tier (eval tier:<name>),
    # where an answer from another tier would be discarded anyway.
    no_fallback: bool = False
    # execute only: write a second program independently and compare the two outputs
    # (execute.run_cross_check). None: use execute.cross_check from the config.
    cross_check: bool | None = None
    # execute only: the tier that writes program B. None: execute.cross_check_tier, else
    # the tier that writes program A. Setting it turns the cross-check on.
    cross_check_tier: str | None = None
    # auto without sources and without a forced tier only: run the computation check
    # before the closed-book call. None: use auto.computation_check from the config.
    computation_check: bool | None = None
    # codetest only: the tier that writes the tests. None: code_test.tests_tier.
    tests_tier: str | None = None
    # mcp only, and required there: the folder the documents server serves for this run.
    docs_root: str | None = None

    @model_validator(mode="after")
    def _options_match_the_mode(self) -> "AskRequest":
        if self.mode in SOURCE_MODES and not self.sources:
            raise ValueError(f"mode '{self.mode}' needs at least one source")
        if self.mode not in (*SOURCE_MODES, "auto") and self.sources:
            raise ValueError("sources need mode 'auto', 'openbook' or 'stuffed'")
        if self.require_citations and not self.sources:
            raise ValueError("require_citations needs sources: without them there is nothing to cite")
        if self.require_citations and self.mode not in ("auto", "openbook"):
            raise ValueError("require_citations needs mode 'auto' or 'openbook'")
        if self.simulate == "triage-timeout" and self.mode == "auto":
            raise ValueError("simulate 'triage-timeout' needs a mode with a triage call (route, chain or stuffed)")
        if self.simulate in CHAIN_ONLY_SIMULATIONS and self.mode != "chain":
            raise ValueError(f"simulate '{self.simulate}' needs mode 'chain'")
        if self.mode == "execute" and self.simulate not in (None, "answer-error"):
            raise ValueError("mode 'execute' supports simulate 'answer-error' only")
        if self.mode == "codetest" and self.simulate not in (None, "answer-error"):
            raise ValueError("mode 'codetest' supports simulate 'answer-error' only")
        if self.mode != "codetest" and self.tests_tier is not None:
            raise ValueError("tests_tier needs mode 'codetest'")
        if self.mode != "execute" and (self.cross_check is not None or self.cross_check_tier is not None):
            raise ValueError("cross_check and cross_check_tier need mode 'execute'")
        if self.cross_check is False and self.cross_check_tier is not None:
            raise ValueError("cross_check_tier cannot be set when cross_check is false")
        if self.computation_check is not None and (
            self.mode != "auto" or self.sources or self.tier is not None
        ):
            raise ValueError("computation_check needs mode 'auto' without sources and without a forced tier")
        if self.simulate in OPENBOOK_ONLY_SIMULATIONS and self.mode not in GROUNDED_MODES:
            raise ValueError(f"simulate '{self.simulate}' needs mode 'openbook' or 'mcp'")
        if self.simulate in MCP_ONLY_SIMULATIONS and self.mode != "mcp":
            raise ValueError(f"simulate '{self.simulate}' needs mode 'mcp'")
        if self.mode == "mcp" and not self.docs_root:
            raise ValueError("mode 'mcp' needs docs_root: the folder of documents to serve")
        if self.mode != "mcp" and self.docs_root is not None:
            raise ValueError("docs_root needs mode 'mcp'")
        if self.mode == "mcp" and self.simulate in ("triage-timeout", *CHAIN_ONLY_SIMULATIONS):
            raise ValueError(f"simulate '{self.simulate}' does not apply to mode 'mcp'")
        verifier_allowed = self.mode in ("chain", *GROUNDED_MODES) or (self.mode == "auto" and self.sources)
        if self.verifier_tier is not None and not verifier_allowed:
            raise ValueError("verifier_tier needs mode 'chain', 'openbook' or 'mcp', or sources")
        return self

    @classmethod
    def from_question(cls, question: str, **overrides: Any) -> "AskRequest":
        return cls(messages=[Message(role="user", content=question)], **overrides)


class TriageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: RouteLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ComputationCheckDecision(BaseModel):
    """What the computation check tier must return."""

    model_config = ConfigDict(extra="forbid")

    # Declared first for the same reason as DraftAnswer.working.
    reason: str
    # True when a short standard-library Python program can compute the answer.
    computable: bool
    confidence: float = Field(ge=0.0, le=1.0)


class DraftAnswer(BaseModel):
    """What a draft or repair model must return in chain mode."""

    model_config = ConfigDict(extra="forbid")

    # Declared first so the model writes its working before it commits to an answer.
    # Measured on the golden set: with the answer first, the local model got arithmetic
    # and code-tracing questions wrong that it answered correctly in plain text.
    working: str = ""
    answer: str = Field(min_length=1)
    # The factual or computed claims the answer rests on.
    key_claims: list[str]
    assumptions: list[str]


class VerifierVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["supported", "unsupported", "uncertain"]
    critique: str
    failed_claims: list[str]


class Claim(BaseModel):
    """One statement of an open-book answer and the passage text it rests on."""

    model_config = ConfigDict(extra="forbid")

    text: str
    passage_id: str
    # Copied verbatim from the cited passage; checked in code before any verifier runs.
    quote: str


class OpenbookDraft(BaseModel):
    """What a draft or escalation model must return in openbook mode."""

    model_config = ConfigDict(extra="forbid")

    # Declared first for the same reason as DraftAnswer.working.
    working: str = ""
    # False when the passages do not contain the answer.
    answerable: bool
    answer: str
    claims: list[Claim]

    @model_validator(mode="after")
    def _an_answerable_draft_has_an_answer(self) -> "OpenbookDraft":
        if self.answerable and not self.answer.strip():
            raise ValueError("answer must not be empty when answerable is true")
        return self


class ExecuteDraft(BaseModel):
    """What the answering model must return in execute mode."""

    model_config = ConfigDict(extra="forbid")

    # Declared first for the same reason as DraftAnswer.working.
    working: str = ""
    # Python 3 source, standard library only, that prints the answer as its last line.
    program: str = Field(min_length=1)
    # What the model says the program's last line of output will be, or None when it
    # declines to say: the prompt tells it not to work a calculation out by hand. A blank
    # string counts as no claim.
    claimed_answer: str | None = None

    @model_validator(mode="after")
    def _blank_claim_is_no_claim(self) -> "ExecuteDraft":
        if self.claimed_answer is not None and not self.claimed_answer.strip():
            self.claimed_answer = None
        return self


class CodeInterface(BaseModel):
    """What the interface call must return in codetest mode: the one function both the
    solution and the tests are written against. Checked in code (codetest.check_interface)."""

    model_config = ConfigDict(extra="forbid")

    function_name: str = Field(min_length=1)
    # The function's `def` line, for example `def luhn_check_digit(payload: str) -> int:`.
    signature: str = Field(min_length=1)


class SolutionDraft(BaseModel):
    """What the solution call and the repair call must return in codetest mode."""

    model_config = ConfigDict(extra="forbid")

    # Declared first for the same reason as DraftAnswer.working.
    working: str = ""
    # Python 3 source of the module `solution`, standard library only.
    solution: str = Field(min_length=1)


class UnitTestsDraft(BaseModel):
    """What the test-writing call must return in codetest mode."""

    model_config = ConfigDict(extra="forbid")

    # Declared first for the same reason as DraftAnswer.working.
    working: str = ""
    # Python 3 source of a `unittest` module that imports the function from `solution`.
    tests: str = Field(min_length=1)


class SandboxRunSummary(BaseModel):
    """One program run in the sandbox as shown to the caller (mirrors `sandbox_runs`)."""

    seq: int
    attempt: Literal["first", "repair"]
    sandbox_name: str | None
    status: str
    error_type: str | None
    exit_code: int | None
    timed_out: bool
    duration_ms: float
    total_ms: float
    network_policy: str
    environment_names: list[str]
    sandbox_removed: bool
    # agree | disagree | no_claim | program_failed | infra_failed; in a cross-check
    # ran | program_failed | infra_failed; in codetest mode passed | vacuous | tests_failed |
    # solution_invalid | tests_invalid | not_finished | infra_failed
    outcome: str
    # None when the model gave no claim for this program.
    claimed_answer: str | None
    # codetest mode: the result line the test run printed (JSON with the counts), when it
    # printed a valid one.
    executed_answer: str | None
    # Which program of a cross-check this run belongs to: a | b. "a" outside a cross-check.
    program_label: str = "a"


class Citation(BaseModel):
    """A claim of the returned answer that passed the checks, as shown to the caller."""

    source: str
    passage_id: str
    quote: str
    # mcp mode only: the SHA-256 the documents server gave for the cited document, and the
    # `tool_calls` rows (by seq) whose results make up its text.
    document_sha256: str | None = None
    tool_call_seqs: list[int] = Field(default_factory=list)


class PlannedCall(BaseModel):
    """One tool call a planning model asks for in mcp mode."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    """What the planning model must return in mcp mode: the tool calls it asks for.

    Each call is a tool name and an arguments object, nothing more. Nothing here is
    trusted: every call goes through toolgate.decide_call before it is sent."""

    model_config = ConfigDict(extra="forbid")

    # A plan longer than this is not a plan: it fails the schema and is treated as invalid.
    calls: list[PlannedCall] = Field(max_length=PLAN_SCHEMA_MAX_CALLS)


class ToolCallSummary(BaseModel):
    """One tool call as shown to the caller (mirrors `tool_calls`). No result text."""

    seq: int
    origin: Literal["code", "plan"]
    tool: str
    arguments: str
    status: Literal["ok", "refused", "error"]
    error_type: str | None
    error_message: str | None
    result_bytes: int | None
    result_sha256: str | None
    duration_ms: float


class CallResult(BaseModel):
    """What every client returns for one model call, success or failure."""

    text: str = ""
    status: Literal["ok", "error"]
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    cost_source: CostSource | None = None
    ollama_load_duration_ms: float | None = None
    ollama_total_duration_ms: float | None = None
    ollama_prompt_eval_ms: float | None = None
    ollama_eval_ms: float | None = None
    # Why the model stopped. `finish_reason` is the provider-normalised value (OpenRouter:
    # stop | length | content_filter | tool_calls | error; Ollama: done_reason).
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    reasoning_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class TriageOutcome(BaseModel):
    status: TriageStatus
    decision: TriageDecision | None = None
    call: CallResult | None = None


class ComputationCheckOutcome(BaseModel):
    status: TriageStatus
    decision: ComputationCheckDecision | None = None
    call: CallResult | None = None


class CallSummary(BaseModel):
    """One model call as shown to the caller (mirrors the telemetry CallRecord)."""

    seq: int
    role: CallRole
    tier: str
    provider: str
    model: str
    effort: str | None
    status: str
    error_type: str | None
    error_message: str | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    cost_source: str | None
    ollama_load_duration_ms: float | None


class RunResult(BaseModel):
    trace_id: str
    status: Literal["ok", "error"]
    answer: str | None
    error_type: str | None = None
    error_message: str | None = None
    triage_status: TriageStatus
    triage_label: str | None = None
    triage_confidence: float | None = None
    triage_reason: str | None = None
    route_reason: RouteReason
    selected_tier: str
    final_tier: str
    final_model: str
    effort: str | None = None
    fallback_used: bool = False
    fallback_skipped_reason: str | None = None
    # The mode that ran. An auto request reports the mode auto picked, with `auto_reason`.
    mode: EffectiveMode = "route"
    verification_status: VerificationStatus = "skipped"
    verification_critique: str | None = None
    repaired: bool = False
    schema_escalated: bool = False
    verifier_tier: str | None = None
    # Set when a repair (chain) or escalation (openbook) was attempted and failed; the
    # answer is then the unrepaired draft.
    repair_error_type: str | None = None
    # Open-book fields. In route and chain mode: empty, "skipped", False, 0. In openbook
    # mode `repaired` means "escalated", and grounding_status is the verdict on the first
    # draft: an escalated answer has passed the deterministic quote check only.
    citations: list[Citation] = Field(default_factory=list)
    grounding_status: GroundingStatus = "skipped"
    # The pipeline concluded the sources do not contain the answer.
    abstained: bool = False
    passages_retrieved: int = 0
    sources_count: int = 0
    # hybrid | embedding | lexical | all (corpus fit the budget, retrieval skipped) | none (stuffed)
    retrieval_method: str | None = None
    # Why embeddings were not used when retrieval_method is lexical.
    retrieval_note: str | None = None
    # Claims of the returned answer, and how many of them passed the checks.
    claims_total: int = 0
    claims_supported: int = 0
    grounding_verifier_tier: str | None = None
    # One line per claim that did not pass, and the answerability judgment.
    grounding_notes: list[str] = Field(default_factory=list)
    # What the escalation call was sent (passage count, estimated tokens of passage
    # text); None when no escalation was attempted.
    escalation_passages: int | None = None
    escalation_tokens_estimated: int | None = None
    # Why auto mode picked `mode`; None when the caller picked the mode.
    auto_reason: str | None = None
    # Auto mode's computation check: its verdict, the tier asked, and the confidence and
    # reason it returned (None when it did not run or its output was not usable).
    computation_verdict: ComputationVerdict = "not_run"
    computation_check_tier: str | None = None
    computation_check_confidence: float | None = None
    computation_check_reason: str | None = None
    # Execute mode. "skipped" in every other mode. After a repair, `repaired` is true and
    # these describe the repaired program. One exception: when auto mode sent the question
    # to execute mode and execution failed outright, the answer is one closed-book call;
    # `mode` is then "route", this is "failed" and `execution_note` says the answer was not
    # verified.
    execution_status: ExecutionStatus = "skipped"
    # The answer the model claimed, and the last line the program printed (None when no
    # program ran cleanly).
    claimed_answer: str | None = None
    executed_answer: str | None = None
    # One line saying what was and was not verified by execution.
    execution_note: str | None = None
    # What the returned program printed before its answer line: the explanation, as executed.
    # A disputed cross-check carries both programs' explanations, each under its label.
    executed_explanation: str | None = None
    # Cross-check. `executed_answer` is then program A's output and `executed_answer_b`
    # program B's (None for a program that did not run cleanly). `cross_check_tier` wrote
    # program B. `cross_check_rule` is how the outputs were compared: integer | hex |
    # decimal | number_list | string; None when fewer than two programs ran.
    cross_check: bool = False
    cross_check_tier: str | None = None
    executed_answer_b: str | None = None
    cross_check_rule: str | None = None
    # Codetest mode. `execution_status` is then tests_passed | tests_failed | untested |
    # failed | skipped and `execution_note` says what the tests did and did not show.
    # `code_function` and `code_signature` are the interface both writers were given.
    # `code_solution` is the solution in the answer, `code_first_solution` the first draft
    # (the same text unless `repaired`), `code_tests` the generated tests (None when none
    # were written). `tests_run` and `tests_failed` count the returned solution's test run,
    # skipped tests left out; None when it had no run with a result. `code_first_outcome`
    # is how the first solution's run was classed (see codetest.classify_test_run), kept
    # apart from the repaired one's.
    code_function: str | None = None
    code_signature: str | None = None
    code_solution: str | None = None
    code_first_solution: str | None = None
    code_tests: str | None = None
    tests_tier: str | None = None
    tests_run: int | None = None
    tests_failed: int | None = None
    code_first_outcome: str | None = None
    sandbox_runs: list[SandboxRunSummary] = Field(default_factory=list)
    # mcp mode. `tool_calls` is every call the gate saw, sent or refused, in order.
    # `plan_status` says what became of the model's tool plan and `plan_tier` who wrote it.
    # `documents_listed` and `documents_read` count the server's listing and the documents
    # whose text reached the grounded pipeline. `tool_server_stopped` is whether the server
    # process was gone before the first draft call (None: not known, or it never started).
    # Outside mcp mode: empty, None and 0.
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)
    plan_status: PlanStatus | None = None
    plan_tier: str | None = None
    docs_root: str | None = None
    documents_listed: int = 0
    documents_read: int = 0
    tool_server_stopped: bool | None = None
    tool_server_environment: list[str] = Field(default_factory=list)
    total_latency_ms: float
    total_cost_usd: float
    calls: list[CallSummary]
