"""Deterministic router: one question in, one answer out, no loops.

Routing is plain if/else, first match wins. In route mode the answer call gets at most
one fallback hop. In chain mode the selected tier writes a draft that goes through the
verification chain in chain.py. In openbook mode the configured drafter answers from
the caller's sources through the grounded pipeline in openbook.py; stuffed mode is its
one-call baseline. In execute mode the configured tier writes a program that runs in a
sandbox (execute.py, sandbox.py). A failed run returns an explicit error; an answer is
never invented.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from inference_arbiter import auto, chain, execute, openbook, sandbox
from inference_arbiter.clients import ERROR_TIMEOUT, ChatCall, ChatClient, build_clients
from inference_arbiter.config import (
    DEFAULT_CONFIG_PATH,
    HarnessConfig,
    TierConfig,
    load_config,
    load_env_file,
)
from inference_arbiter.schemas import (
    ERROR_SIMULATED_ERROR,
    ERROR_SIMULATED_TIMEOUT,
    STATUS_ERROR,
    STATUS_OK,
    AskRequest,
    CallResult,
    CallSummary,
    Citation,
    ExecutionStatus,
    GroundingStatus,
    Message,
    RouteReason,
    RunResult,
    SandboxRunSummary,
    TriageOutcome,
    VerificationStatus,
)
from inference_arbiter.telemetry import CallRecord, RunRecord, SandboxRunRecord, Telemetry
from inference_arbiter.triage import extract_text, last_user_text, run_triage

DEFAULT_DB_PATH = Path("data/telemetry.db")

ERROR_UNKNOWN_TIER = "unknown_tier"

SKIP_NO_FALLBACK_CONFIGURED = "no_on_error_target"
SKIP_CLOUD_FALLBACK_DISABLED = "cloud_fallback_disabled"
SKIP_FALLBACK_NOT_REQUESTED = "no_fallback_requested"

ROLE_TRIAGE = "triage"
ROLE_ANSWER = "answer"


class UnknownTierError(ValueError):
    """Raised when a caller forces a tier that is not an answer tier in the config."""


class ChainNotConfiguredError(ValueError):
    """Raised when chain mode is requested but the config has no chain block."""


class OpenbookNotConfiguredError(ValueError):
    """Raised when openbook or stuffed mode is requested but the config has no openbook block."""


class ExecuteNotConfiguredError(ValueError):
    """Raised when execute mode is requested but the config has no execute block."""


@dataclass
class _RunOutcome:
    """What either mode hands to the shared telemetry and result code."""

    status: str
    answer: str | None
    error_type: str | None
    error_message: str | None
    final_tier: str
    final_effort: str | None
    fallback_used: bool
    fallback_skipped_reason: str | None
    verification_status: VerificationStatus = "skipped"
    verification_critique: str | None = None
    repaired: bool = False
    schema_escalated: bool = False
    verifier_tier: str | None = None
    repair_error_type: str | None = None
    citations: list[Citation] = field(default_factory=list)
    grounding_status: GroundingStatus = "skipped"
    grounding_verifier_tier: str | None = None
    grounding_notes: list[str] = field(default_factory=list)
    abstained: bool = False
    passages_retrieved: int = 0
    sources_count: int = 0
    retrieval_method: str | None = None
    retrieval_note: str | None = None
    claims_total: int = 0
    claims_supported: int = 0
    escalation_passages: int | None = None
    escalation_tokens_estimated: int | None = None
    execution_status: ExecutionStatus = "skipped"
    claimed_answer: str | None = None
    executed_answer: str | None = None
    execution_note: str | None = None
    executed_explanation: str | None = None
    cross_check: bool = False
    cross_check_tier: str | None = None
    executed_answer_b: str | None = None
    cross_check_rule: str | None = None
    sandbox_records: list[SandboxRunRecord] = field(default_factory=list)
    call_records: list[CallRecord] = field(default_factory=list)
    call_messages: list[str | None] = field(default_factory=list)


def choose_route(
    config: HarnessConfig, forced_tier: str | None, triage: TriageOutcome
) -> tuple[str, RouteReason]:
    """Return (selected_tier, route_reason). First match wins."""
    if forced_tier is not None:
        return forced_tier, "forced"

    if triage.status == "timeout":
        return config.fallback_tier, "triage_timeout"
    if triage.status == "error":
        return config.fallback_tier, "triage_error"
    if triage.status == "invalid" or triage.decision is None:
        return config.fallback_tier, "triage_invalid"

    label = triage.decision.label
    if triage.decision.confidence < config.confidence_threshold:
        # One step up. A label with no escalation entry (frontier) stays where it is.
        return config.escalation.get(label, label), "low_confidence"

    return label, "triage_label"


def resolve_fallback_tier(config: HarnessConfig, failed_tier: str) -> tuple[str | None, str | None]:
    """Return (fallback_tier, skipped_reason) for a failed answer tier."""
    fallback_tier = config.tiers[failed_tier].on_error
    if fallback_tier is None:
        return None, SKIP_NO_FALLBACK_CONFIGURED
    if config.tiers[fallback_tier].cloud and not config.allow_cloud_fallback:
        return None, SKIP_CLOUD_FALLBACK_DISABLED
    return fallback_tier, None


def text_only_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Phase 1 answers from text only: non-text content parts are not forwarded."""
    return [{"role": message.role, "content": extract_text(message.content)} for message in messages]


def build_answer_call(
    tier: TierConfig,
    messages: list[dict[str, str]],
    effort_override: str | None,
    json_schema: dict[str, Any] | None = None,
    think_override: bool | str | None = None,
) -> ChatCall:
    effort = effort_override if effort_override is not None else tier.effort
    if think_override is not None:
        think_setting: bool | str | None = think_override
    elif tier.think == "effort":
        think_setting = effort
    else:
        think_setting = tier.think
    return ChatCall(
        model=tier.model,
        messages=messages,
        timeout_s=tier.timeout_s,
        effort=effort,
        think=think_setting,
        num_ctx=tier.num_ctx,
        keep_alive=tier.keep_alive,
        temperature=tier.temperature,
        max_output_tokens=tier.max_output_tokens,
        json_schema=json_schema,
        input_price_per_mtok=tier.input_price_per_mtok,
        output_price_per_mtok=tier.output_price_per_mtok,
    )


def _simulated_failure(error_type: str) -> CallResult:
    return CallResult(
        status=STATUS_ERROR,
        error_type=error_type,
        error_message="failure injected by --simulate; no model was called",
        latency_ms=0.0,
    )


def _call_record_status(result: CallResult, validation_status: str | None) -> str:
    if validation_status == "invalid":
        return "invalid"
    if result.ok:
        return STATUS_OK
    if result.error_type in (ERROR_TIMEOUT, ERROR_SIMULATED_TIMEOUT):
        return "timeout"
    return STATUS_ERROR


class Harness:
    def __init__(
        self,
        config: HarnessConfig,
        clients: dict[str, ChatClient],
        telemetry: Telemetry,
        source_transport: httpx.BaseTransport | None = None,
        sandbox_runner: sandbox.CommandRunner | None = None,
    ) -> None:
        self._config = config
        self._clients = clients
        self._telemetry = telemetry
        # Only tests set this: it replaces the `sbx` subprocess in execute mode.
        self._sandbox_runner = sandbox_runner
        # Only tests set this: it replaces the network when a source is a URL.
        self._source_transport = source_transport

    def _make_call_record(
        self,
        seq: int,
        role: str,
        tier_name: str,
        effort: str | None,
        result: CallResult,
        validation_status: str | None = None,
    ) -> CallRecord:
        tier = self._config.tiers[tier_name]
        return CallRecord(
            seq=seq,
            role=role,
            tier=tier_name,
            provider=tier.provider,
            model=tier.model,
            effort=effort,
            timeout_s=tier.timeout_s,
            status=_call_record_status(result, validation_status),
            error_type=result.error_type,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            cost_usd=result.cost_usd,
            cost_source=result.cost_source,
            ollama_load_duration_ms=result.ollama_load_duration_ms,
            ollama_total_duration_ms=result.ollama_total_duration_ms,
            ollama_prompt_eval_ms=result.ollama_prompt_eval_ms,
            ollama_eval_ms=result.ollama_eval_ms,
            finish_reason=result.finish_reason,
            native_finish_reason=result.native_finish_reason,
            reasoning_tokens=result.reasoning_tokens,
        )

    def _triage(self, request: AskRequest) -> TriageOutcome:
        if request.tier is not None:
            return TriageOutcome(status="skipped")
        if request.simulate == "triage-timeout":
            return TriageOutcome(status="timeout", call=_simulated_failure(ERROR_SIMULATED_TIMEOUT))
        triage_client = self._clients[self._config.tiers[self._config.triage_tier].provider]
        return run_triage(self._config, triage_client, request.messages)

    def _call_answer_tier(
        self,
        tier_name: str,
        messages: list[dict[str, str]],
        effort_override: str | None,
        inject_failure: bool,
        json_schema: dict[str, Any] | None = None,
        think_override: bool | str | None = None,
    ) -> tuple[CallResult, str | None]:
        tier = self._config.tiers[tier_name]
        answer_call = build_answer_call(tier, messages, effort_override, json_schema, think_override)
        if inject_failure:
            return _simulated_failure(ERROR_SIMULATED_ERROR), answer_call.effort
        return self._clients[tier.provider].chat(answer_call), answer_call.effort

    def _answer_route(
        self, request: AskRequest, selected_tier: str, answer_messages: list[dict[str, str]]
    ) -> _RunOutcome:
        call_records: list[CallRecord] = []
        call_messages: list[str | None] = []
        final_tier = selected_tier
        fallback_used = False
        fallback_skipped_reason: str | None = None

        final_result, final_effort = self._call_answer_tier(
            selected_tier,
            answer_messages,
            request.effort,
            inject_failure=request.simulate == "answer-error",
        )
        call_records.append(self._make_call_record(0, ROLE_ANSWER, selected_tier, final_effort, final_result))
        call_messages.append(final_result.error_message)

        if not final_result.ok:
            if request.no_fallback:
                fallback_tier, fallback_skipped_reason = None, SKIP_FALLBACK_NOT_REQUESTED
            else:
                fallback_tier, fallback_skipped_reason = resolve_fallback_tier(self._config, selected_tier)
            if fallback_tier is not None:
                # The single fallback hop. Its result is final, success or failure.
                fallback_used = True
                final_tier = fallback_tier
                final_result, final_effort = self._call_answer_tier(
                    fallback_tier, answer_messages, request.effort, inject_failure=False
                )
                call_records.append(
                    self._make_call_record(0, ROLE_ANSWER, fallback_tier, final_effort, final_result)
                )
                call_messages.append(final_result.error_message)

        return _RunOutcome(
            status=STATUS_OK if final_result.ok else STATUS_ERROR,
            answer=final_result.text if final_result.ok else None,
            error_type=final_result.error_type,
            error_message=final_result.error_message,
            final_tier=final_tier,
            final_effort=final_effort,
            fallback_used=fallback_used,
            fallback_skipped_reason=fallback_skipped_reason if not final_result.ok else None,
            call_records=call_records,
            call_messages=call_messages,
        )

    def _answer_chain(
        self,
        request: AskRequest,
        selected_tier: str,
        answer_messages: list[dict[str, str]],
        verifier_tier_name: str,
    ) -> _RunOutcome:
        config = self._config

        def call_tier(
            tier_name: str,
            messages: list[dict[str, str]],
            json_schema: dict[str, Any] | None,
            effort_override: str | None,
        ) -> tuple[CallResult, str | None]:
            return self._call_answer_tier(
                tier_name, messages, effort_override, inject_failure=False, json_schema=json_schema
            )

        chain_outcome = chain.run_chain(
            config=config,
            chain_config=config.chain,
            clients=self._clients,
            request=request,
            question=last_user_text(request.messages),
            conversation=answer_messages,
            draft_tier=selected_tier,
            verifier_tier_name=verifier_tier_name,
            call_tier=call_tier,
            simulated_answer_failure=lambda: _simulated_failure(ERROR_SIMULATED_ERROR),
            resolve_fallback=lambda failed_tier: resolve_fallback_tier(config, failed_tier),
        )

        call_records: list[CallRecord] = []
        call_messages: list[str | None] = []
        for step in chain_outcome.steps:
            call_records.append(
                self._make_call_record(
                    0, step.role, step.tier, step.effort, step.result, "invalid" if step.invalid else None
                )
            )
            call_messages.append(step.result.error_message)

        run_failed = chain_outcome.status != STATUS_OK
        return _RunOutcome(
            status=chain_outcome.status,
            answer=chain_outcome.answer,
            error_type=chain_outcome.error_type,
            error_message=chain_outcome.error_message,
            final_tier=chain_outcome.final_tier,
            final_effort=chain_outcome.final_effort,
            fallback_used=chain_outcome.fallback_used,
            fallback_skipped_reason=chain_outcome.fallback_skipped_reason if run_failed else None,
            verification_status=chain_outcome.verification_status,
            verification_critique=chain_outcome.verification_critique,
            repaired=chain_outcome.repaired,
            schema_escalated=chain_outcome.schema_escalated,
            verifier_tier=chain_outcome.verifier_tier,
            repair_error_type=chain_outcome.repair_error_type,
            call_records=call_records,
            call_messages=call_messages,
        )

    def _outcome_from_openbook(self, openbook_outcome: openbook.OpenbookOutcome) -> _RunOutcome:
        call_records: list[CallRecord] = []
        call_messages: list[str | None] = []
        for step in openbook_outcome.steps:
            call_records.append(
                self._make_call_record(
                    0, step.role, step.tier, step.effort, step.result, "invalid" if step.invalid else None
                )
            )
            call_messages.append(step.result.error_message)

        run_failed = openbook_outcome.status != STATUS_OK
        return _RunOutcome(
            status=openbook_outcome.status,
            answer=openbook_outcome.answer,
            error_type=openbook_outcome.error_type,
            error_message=openbook_outcome.error_message,
            final_tier=openbook_outcome.final_tier,
            final_effort=openbook_outcome.final_effort,
            fallback_used=openbook_outcome.fallback_used,
            fallback_skipped_reason=openbook_outcome.fallback_skipped_reason if run_failed else None,
            repaired=openbook_outcome.escalated,
            schema_escalated=openbook_outcome.schema_escalated,
            repair_error_type=openbook_outcome.escalation_error_type,
            citations=openbook_outcome.citations,
            grounding_status=openbook_outcome.grounding_status,
            grounding_verifier_tier=openbook_outcome.grounding_verifier_tier,
            grounding_notes=openbook_outcome.grounding_notes,
            abstained=openbook_outcome.abstained,
            passages_retrieved=openbook_outcome.passages_retrieved,
            sources_count=openbook_outcome.sources_count,
            retrieval_method=openbook_outcome.retrieval_method,
            retrieval_note=openbook_outcome.retrieval_note,
            claims_total=openbook_outcome.claims_total,
            claims_supported=openbook_outcome.claims_supported,
            escalation_passages=openbook_outcome.escalation_passages,
            escalation_tokens_estimated=openbook_outcome.escalation_tokens_estimated,
            call_records=call_records,
            call_messages=call_messages,
        )

    def _plain_tier_caller(self, think_override: bool | str | None = None) -> chain.TierCaller:
        def call_tier(
            tier_name: str,
            messages: list[dict[str, str]],
            json_schema: dict[str, Any] | None,
            effort_override: str | None,
        ) -> tuple[CallResult, str | None]:
            return self._call_answer_tier(
                tier_name,
                messages,
                effort_override,
                inject_failure=False,
                json_schema=json_schema,
                think_override=think_override,
            )

        return call_tier

    def _resolve_auto(
        self, request: AskRequest
    ) -> tuple[auto.AutoDecision, openbook.PreloadedSources | None]:
        """Pick the mode for an auto request. Sources are loaded here, once, and reused."""
        config = self._config
        if not request.sources:
            return auto.decide_mode(False, 0, None, config.auto.stuffed_fill_ratio, False), None
        if config.openbook is None:
            raise OpenbookNotConfiguredError("sources need an `openbook:` block in the config")
        preloaded = openbook.preload_sources(config.openbook, request.sources, self._source_transport)
        if preloaded.error is not None:
            # The pipeline reports the load failure; no model is called.
            return auto.AutoDecision("openbook", auto.REASON_LOAD_FAILED), preloaded
        answering_tier = request.tier or config.auto_closed_book_tier()
        decision = auto.decide_mode(
            True,
            openbook.estimate_stuffed_tokens(last_user_text(request.messages), preloaded.documents),
            config.tiers[answering_tier].context_window_tokens,
            config.auto.stuffed_fill_ratio,
            request.require_citations,
        )
        return decision, preloaded

    def _answer_openbook(
        self,
        request: AskRequest,
        selected_tier: str,
        verifier_tier_name: str,
        preloaded: openbook.PreloadedSources | None = None,
    ) -> _RunOutcome:
        config = self._config
        openbook_outcome = openbook.run_openbook(
            config=config,
            openbook_config=config.openbook,
            clients=self._clients,
            request=request,
            question=last_user_text(request.messages),
            draft_tier=selected_tier,
            verifier_tier_name=verifier_tier_name,
            call_tier=self._plain_tier_caller(),
            simulated_answer_failure=lambda: _simulated_failure(ERROR_SIMULATED_ERROR),
            resolve_fallback=lambda failed_tier: resolve_fallback_tier(config, failed_tier),
            transport=self._source_transport,
            preloaded=preloaded,
        )
        return self._outcome_from_openbook(openbook_outcome)

    def _answer_stuffed(
        self,
        request: AskRequest,
        selected_tier: str,
        preloaded: openbook.PreloadedSources | None = None,
    ) -> _RunOutcome:
        stuffed_outcome = openbook.run_stuffed(
            openbook_config=self._config.openbook,
            request=request,
            question=last_user_text(request.messages),
            tier_name=selected_tier,
            call_tier=self._plain_tier_caller(),
            simulated_answer_failure=lambda: _simulated_failure(ERROR_SIMULATED_ERROR),
            default_effort=self._config.tiers[selected_tier].effort,
            transport=self._source_transport,
            context_window_tokens=self._config.tiers[selected_tier].context_window_tokens,
            preloaded=preloaded,
        )
        return self._outcome_from_openbook(stuffed_outcome)

    def _answer_execute(
        self, request: AskRequest, selected_tier: str, answer_messages: list[dict[str, str]]
    ) -> _RunOutcome:
        execute_config = self._config.execute
        limits = sandbox.SandboxLimits(
            timeout_s=execute_config.timeout_s,
            max_output_bytes=execute_config.max_output_bytes,
            max_program_bytes=execute_config.max_program_bytes,
            cpus=execute_config.cpus,
            memory=execute_config.memory,
            create_timeout_s=execute_config.create_timeout_s,
            sbx_path=execute_config.sbx_path,
            network_canary=execute_config.network_canary,
        )

        def run_in_sandbox(program: str) -> sandbox.SandboxResult:
            return sandbox.run_program(
                program, sandbox.RUNNER_PYTHON, limits, command_runner=self._sandbox_runner
            )

        simulated_failure = None
        if request.simulate == "answer-error":
            simulated_failure = lambda: _simulated_failure(ERROR_SIMULATED_ERROR)  # noqa: E731
        cross_check_tier = self._resolve_cross_check_tier(request, selected_tier)
        if cross_check_tier is None:
            execute_outcome = execute.run_execute(
                question=last_user_text(request.messages),
                conversation=answer_messages,
                tier_name=selected_tier,
                max_repairs=execute_config.max_repairs,
                effort_override=request.effort,
                call_tier=self._plain_tier_caller(think_override=execute_config.think),
                run_in_sandbox=run_in_sandbox,
                simulated_answer_failure=simulated_failure,
            )
        else:
            tiers = self._config.tiers
            think_b = execute_config.cross_check_think
            if think_b is None and cross_check_tier == selected_tier:
                think_b = execute_config.think
            execute_outcome = execute.run_cross_check(
                question=last_user_text(request.messages),
                conversation=answer_messages,
                tier_a=selected_tier,
                tier_b=cross_check_tier,
                writer_a=f"tier {selected_tier}, model {tiers[selected_tier].model}",
                writer_b=f"tier {cross_check_tier}, model {tiers[cross_check_tier].model}",
                # One model writing both programs shares its own habits: ask B for another method.
                independent_method=tiers[selected_tier].model == tiers[cross_check_tier].model,
                max_repairs=execute_config.max_repairs,
                effort_override=request.effort,
                call_tier_a=self._plain_tier_caller(think_override=execute_config.think),
                call_tier_b=self._plain_tier_caller(think_override=think_b),
                run_in_sandbox=run_in_sandbox,
                simulated_answer_failure=simulated_failure,
                parallel=execute_config.cross_check_parallel,
            )

        call_records: list[CallRecord] = []
        call_messages: list[str | None] = []
        for step in execute_outcome.steps:
            call_records.append(
                self._make_call_record(
                    0, step.role, step.tier, step.effort, step.result, "invalid" if step.invalid else None
                )
            )
            call_messages.append(step.result.error_message)

        sandbox_records = [
            SandboxRunRecord(
                seq=position + 1,
                attempt=attempt.attempt,
                sandbox_name=attempt.result.sandbox_name,
                runner=attempt.result.runner,
                program=attempt.program,
                claimed_answer=attempt.claimed_answer,
                executed_answer=attempt.executed_answer,
                status=attempt.result.status,
                error_type=attempt.result.error_type,
                exit_code=attempt.result.exit_code,
                stdout=attempt.result.stdout,
                stderr=attempt.result.stderr,
                stdout_truncated=attempt.result.stdout_truncated,
                stderr_truncated=attempt.result.stderr_truncated,
                duration_ms=attempt.result.duration_ms,
                total_ms=attempt.result.total_ms,
                timed_out=attempt.result.timed_out,
                network_policy=attempt.result.network_policy,
                environment_names=json.dumps(list(attempt.result.environment_names)),
                sandbox_removed=attempt.result.sandbox_removed,
                outcome=attempt.outcome,
                program_label=attempt.program_label,
            )
            for position, attempt in enumerate(execute_outcome.attempts)
        ]
        return _RunOutcome(
            status=execute_outcome.status,
            answer=execute_outcome.answer,
            error_type=execute_outcome.error_type,
            error_message=execute_outcome.error_message,
            final_tier=selected_tier,
            final_effort=execute_outcome.final_effort,
            fallback_used=False,
            fallback_skipped_reason=None,
            repaired=execute_outcome.repaired,
            repair_error_type=execute_outcome.repair_error_type,
            execution_status=execute_outcome.execution_status,
            claimed_answer=execute_outcome.claimed_answer,
            executed_answer=execute_outcome.executed_answer,
            execution_note=execute_outcome.execution_note,
            executed_explanation=execute_outcome.executed_explanation,
            cross_check=execute_outcome.cross_check,
            cross_check_tier=execute_outcome.cross_check_tier,
            executed_answer_b=execute_outcome.executed_answer_b,
            cross_check_rule=execute_outcome.cross_check_rule,
            sandbox_records=sandbox_records,
            call_records=call_records,
            call_messages=call_messages,
        )

    def _resolve_cross_check_tier(self, request: AskRequest, selected_tier: str) -> str | None:
        """The tier that writes program B, or None when this run has no cross-check.

        The request wins over the config: `cross_check` switches it, `cross_check_tier`
        names B's tier and switches it on. B's tier defaults to execute.cross_check_tier,
        then to the tier that writes program A.
        """
        execute_config = self._config.execute
        if request.cross_check is None:
            enabled = execute_config.cross_check or request.cross_check_tier is not None
        else:
            enabled = request.cross_check
        if not enabled:
            return None
        return request.cross_check_tier or execute_config.cross_check_tier or selected_tier

    def _resolve_grounding_verifier_tier(self, request: AskRequest) -> str:
        config = self._config
        if request.verifier_tier is None:
            return config.openbook.grounding_verifier_tier
        if request.verifier_tier not in config.verifier_tier_names():
            raise UnknownTierError(
                f"unknown verifier tier '{request.verifier_tier}'; "
                f"choose one of {config.verifier_tier_names()}"
            )
        return request.verifier_tier

    def _resolve_verifier_tier(self, request: AskRequest) -> str:
        config = self._config
        if config.chain is None:
            raise ChainNotConfiguredError("chain mode needs a `chain:` block in the config")
        if request.verifier_tier is None:
            return config.chain.verifier_tier
        if request.verifier_tier not in config.verifier_tier_names():
            raise UnknownTierError(
                f"unknown verifier tier '{request.verifier_tier}'; "
                f"choose one of {config.verifier_tier_names()}"
            )
        return request.verifier_tier

    def answer(self, request: AskRequest) -> RunResult:
        run_started_at = time.monotonic()
        trace_id = uuid.uuid4().hex
        ts_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        config = self._config

        if request.tier is not None and request.tier not in config.answer_tier_names():
            raise UnknownTierError(
                f"unknown tier '{request.tier}'; choose one of {config.answer_tier_names()}"
            )
        # Auto mode is resolved to one of the explicit modes first; from here on the
        # request carries the mode that runs.
        auto_reason: str | None = None
        preloaded: openbook.PreloadedSources | None = None
        if request.mode == "auto":
            auto_decision, preloaded = self._resolve_auto(request)
            auto_reason = auto_decision.reason
            request = request.model_copy(update={"mode": auto_decision.mode})

        verifier_tier_name: str | None = None
        grounding_verifier_name: str | None = None
        if request.mode == "chain":
            verifier_tier_name = self._resolve_verifier_tier(request)
        if request.mode in ("openbook", "stuffed") and config.openbook is None:
            raise OpenbookNotConfiguredError(
                f"{request.mode} mode needs an `openbook:` block in the config"
            )
        if request.mode == "openbook":
            grounding_verifier_name = self._resolve_grounding_verifier_tier(request)
        if request.mode == "execute" and config.execute is None:
            raise ExecuteNotConfiguredError("execute mode needs an `execute:` block in the config")
        if request.cross_check_tier is not None and request.cross_check_tier not in config.answer_tier_names():
            raise UnknownTierError(
                f"cross-check tier '{request.cross_check_tier}' is not an answer tier; "
                f"choose one of {sorted(config.answer_tier_names())}"
            )

        call_records: list[CallRecord] = []
        call_messages: list[str | None] = []

        if request.mode in ("openbook", "execute") or auto_reason is not None:
            # The tier is config-selected: openbook, execute and auto make no triage call.
            triage = TriageOutcome(status="skipped")
        else:
            triage = self._triage(request)
        if triage.call is not None:
            call_records.append(
                self._make_call_record(0, ROLE_TRIAGE, config.triage_tier, None, triage.call, triage.status)
            )
            call_messages.append(triage.call.error_message)

        if request.mode == "openbook" and request.tier is None:
            selected_tier, route_reason = config.openbook.drafter_tier, "configured"
        elif request.mode == "execute" and request.tier is None:
            selected_tier, route_reason = config.execute.tier, "configured"
        elif auto_reason is not None and request.tier is None:
            selected_tier, route_reason = config.auto_closed_book_tier(), "auto"
        else:
            selected_tier, route_reason = choose_route(config, request.tier, triage)
        answer_messages = text_only_messages(request.messages)

        if grounding_verifier_name is not None:
            outcome = self._answer_openbook(request, selected_tier, grounding_verifier_name, preloaded)
        elif request.mode == "stuffed":
            outcome = self._answer_stuffed(request, selected_tier, preloaded)
        elif verifier_tier_name is not None:
            outcome = self._answer_chain(request, selected_tier, answer_messages, verifier_tier_name)
        elif request.mode == "execute":
            outcome = self._answer_execute(request, selected_tier, answer_messages)
        else:
            outcome = self._answer_route(request, selected_tier, answer_messages)

        call_records.extend(outcome.call_records)
        call_messages.extend(outcome.call_messages)
        for position, record in enumerate(call_records):
            record.seq = position + 1

        total_cost_usd = round(sum(record.cost_usd or 0.0 for record in call_records), 8)
        total_latency_ms = round((time.monotonic() - run_started_at) * 1000, 1)
        question_text = last_user_text(request.messages)
        decision = triage.decision

        run_record = RunRecord(
            trace_id=trace_id,
            ts_utc=ts_utc,
            prompt_sha256=hashlib.sha256(question_text.encode("utf-8")).hexdigest(),
            prompt_text=question_text if config.store_prompt_text else None,
            config_sha256=config.sha256,
            triage_status=triage.status,
            triage_label=decision.label if decision else None,
            triage_confidence=decision.confidence if decision else None,
            triage_reason=decision.reason if decision else None,
            route_reason=route_reason,
            selected_tier=selected_tier,
            final_tier=outcome.final_tier,
            final_model=config.tiers[outcome.final_tier].model,
            effort=outcome.final_effort,
            fallback_used=outcome.fallback_used,
            status=outcome.status,
            error_type=outcome.error_type,
            total_latency_ms=total_latency_ms,
            total_cost_usd=total_cost_usd,
            mode=request.mode,
            verification_status=outcome.verification_status,
            verification_critique=outcome.verification_critique,
            repaired=outcome.repaired,
            schema_escalated=outcome.schema_escalated,
            verifier_tier=outcome.verifier_tier,
            sources_count=outcome.sources_count,
            passages_retrieved=outcome.passages_retrieved,
            retrieval_method=outcome.retrieval_method,
            grounding_status=outcome.grounding_status,
            abstained=outcome.abstained,
            claims_total=outcome.claims_total,
            claims_supported=outcome.claims_supported,
            grounding_verifier_tier=outcome.grounding_verifier_tier,
            grounding_notes=json.dumps(outcome.grounding_notes) if outcome.grounding_notes else None,
            escalation_passages=outcome.escalation_passages,
            escalation_tokens_estimated=outcome.escalation_tokens_estimated,
            auto_reason=auto_reason,
            execution_status=outcome.execution_status,
            claimed_answer=outcome.claimed_answer,
            executed_answer=outcome.executed_answer,
            cross_check=outcome.cross_check,
            cross_check_tier=outcome.cross_check_tier,
            executed_answer_b=outcome.executed_answer_b,
            cross_check_rule=outcome.cross_check_rule,
        )
        if outcome.sandbox_records:
            self._telemetry.record(run_record, call_records, outcome.sandbox_records)
        else:
            self._telemetry.record(run_record, call_records)

        call_summaries = [
            CallSummary(
                seq=record.seq,
                role=record.role,
                tier=record.tier,
                provider=record.provider,
                model=record.model,
                effort=record.effort,
                status=record.status,
                error_type=record.error_type,
                error_message=error_message,
                latency_ms=record.latency_ms,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                cost_usd=record.cost_usd,
                cost_source=record.cost_source,
                ollama_load_duration_ms=record.ollama_load_duration_ms,
            )
            for record, error_message in zip(call_records, call_messages)
        ]

        return RunResult(
            trace_id=trace_id,
            status=outcome.status,
            answer=outcome.answer,
            error_type=outcome.error_type,
            error_message=outcome.error_message,
            triage_status=triage.status,
            triage_label=run_record.triage_label,
            triage_confidence=run_record.triage_confidence,
            triage_reason=run_record.triage_reason,
            route_reason=route_reason,
            selected_tier=selected_tier,
            final_tier=outcome.final_tier,
            final_model=run_record.final_model,
            effort=outcome.final_effort,
            fallback_used=outcome.fallback_used,
            fallback_skipped_reason=outcome.fallback_skipped_reason,
            mode=request.mode,
            verification_status=outcome.verification_status,
            verification_critique=outcome.verification_critique,
            repaired=outcome.repaired,
            schema_escalated=outcome.schema_escalated,
            verifier_tier=outcome.verifier_tier,
            repair_error_type=outcome.repair_error_type,
            citations=outcome.citations,
            grounding_status=outcome.grounding_status,
            abstained=outcome.abstained,
            passages_retrieved=outcome.passages_retrieved,
            sources_count=outcome.sources_count,
            retrieval_method=outcome.retrieval_method,
            retrieval_note=outcome.retrieval_note,
            claims_total=outcome.claims_total,
            claims_supported=outcome.claims_supported,
            grounding_verifier_tier=outcome.grounding_verifier_tier,
            grounding_notes=outcome.grounding_notes,
            escalation_passages=outcome.escalation_passages,
            escalation_tokens_estimated=outcome.escalation_tokens_estimated,
            auto_reason=auto_reason,
            execution_status=outcome.execution_status,
            claimed_answer=outcome.claimed_answer,
            executed_answer=outcome.executed_answer,
            execution_note=outcome.execution_note,
            executed_explanation=outcome.executed_explanation,
            cross_check=outcome.cross_check,
            cross_check_tier=outcome.cross_check_tier,
            executed_answer_b=outcome.executed_answer_b,
            cross_check_rule=outcome.cross_check_rule,
            sandbox_runs=[
                SandboxRunSummary(
                    seq=record.seq,
                    attempt=record.attempt,
                    sandbox_name=record.sandbox_name,
                    status=record.status,
                    error_type=record.error_type,
                    exit_code=record.exit_code,
                    timed_out=record.timed_out,
                    duration_ms=record.duration_ms,
                    total_ms=record.total_ms,
                    network_policy=record.network_policy,
                    environment_names=json.loads(record.environment_names),
                    sandbox_removed=record.sandbox_removed,
                    outcome=record.outcome,
                    claimed_answer=record.claimed_answer,
                    executed_answer=record.executed_answer,
                    program_label=record.program_label,
                )
                for record in outcome.sandbox_records
            ],
            total_latency_ms=total_latency_ms,
            total_cost_usd=total_cost_usd,
            calls=call_summaries,
        )


def build_harness(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    env_path: str | Path = ".env",
    triage_tier: str | None = None,
) -> Harness:
    load_env_file(env_path)
    config = load_config(config_path)
    if triage_tier is not None:
        if triage_tier not in config.triage_tier_names():
            raise UnknownTierError(
                f"unknown triage tier '{triage_tier}'; choose one of {config.triage_tier_names()}"
            )
        config.triage_tier = triage_tier
    return Harness(config, build_clients(config), Telemetry(db_path))


def answer(
    request: AskRequest,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> RunResult:
    """Single entry point: build the harness from config and answer one request."""
    return build_harness(config_path, db_path).answer(request)
