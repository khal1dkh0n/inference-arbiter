"""Command line entry point: python -m inference_arbiter ask "question"."""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from inference_arbiter.config import DEFAULT_CONFIG_PATH, ConfigError
from inference_arbiter.router import (
    DEFAULT_DB_PATH,
    ChainNotConfiguredError,
    ExecuteNotConfiguredError,
    OpenbookNotConfiguredError,
    UnknownTierError,
    build_harness,
)
from inference_arbiter.schemas import OPENBOOK_ONLY_SIMULATIONS, AskRequest, RunResult

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference_arbiter", description="Multi-model Q&A router and verification chain")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ask = subcommands.add_parser(
        "ask",
        help="answer one question",
        description=(
            "Without a mode flag the harness picks the strategy (auto): no sources, one call to "
            "auto.closed_book_tier; sources that fit the context window, one stuffed call; sources "
            "that do not fit, the grounded pipeline with citations. With the computation check on, "
            "a question without sources that a short program can compute goes to execute mode. "
            "--route, --chain, --stuffed, --require-citations and --execute pick explicitly."
        ),
    )
    ask.add_argument("question", help="the question text")
    ask.add_argument(
        "--tier",
        help=(
            "answer with this tier, no triage (route_reason: forced); with --chain it forces the "
            "draft tier, with --sources the stuffed or drafting tier"
        ),
    )
    ask.add_argument(
        "--route",
        action="store_true",
        help="route mode: a triage call picks the tier, then one answer call",
    )
    ask.add_argument(
        "--chain",
        action="store_true",
        help="chain mode: triage, draft, schema gate, verify, at most one repair",
    )
    ask.add_argument(
        "--execute",
        action="store_true",
        help=(
            "execute mode: the model writes a short Python program and, optionally, the answer it "
            "claims; the program runs in an isolated sbx sandbox (no network, no host files, no "
            "host environment) and the answer says what execution did and did not confirm"
        ),
    )
    ask.add_argument(
        "--cross-check",
        action="store_true",
        help=(
            "with --execute (implied): a second program is written independently from the question "
            "alone and run in its own sandbox; the answer is `verified` only when both outputs "
            "match, `disputed` when they differ, `one_sided` when only one program ran"
        ),
    )
    ask.add_argument(
        "--cross-check-tier",
        metavar="TIER",
        help=(
            "the tier that writes the second program (implies --cross-check); default "
            "execute.cross_check_tier, else the tier that writes the first"
        ),
    )
    ask.add_argument(
        "--computation-check",
        action="store_true",
        help=(
            "auto mode without sources: first ask auto.computation_check_tier whether a short program "
            "can compute the answer; on a confident yes the question goes to execute mode (with the "
            "cross-check when auto.computation_cross_check is true), otherwise one closed-book call "
            "as usual. Default: config auto.computation_check"
        ),
    )
    ask.add_argument(
        "--no-computation-check",
        action="store_true",
        help="auto mode without sources: do not run the computation check, whatever the config says",
    )
    ask.add_argument(
        "--sources",
        nargs="+",
        metavar="PATH_OR_URL",
        help=(
            "answer from these files or http(s) URLs (put the question before this flag): one "
            "stuffed call when they fit the context window, the grounded pipeline with citations "
            "when they do not"
        ),
    )
    ask.add_argument(
        "--require-citations",
        action="store_true",
        help="with --sources: always use the grounded pipeline with citations, even when the sources fit",
    )
    ask.add_argument(
        "--stuffed",
        action="store_true",
        help=(
            "with --sources: always one call with the full text of every source in the prompt; "
            "triage picks the tier unless --tier is given"
        ),
    )
    ask.add_argument(
        "--verifier-tier",
        help=(
            "with --chain: verifier tier to use (default: config chain.verifier_tier); "
            "with --sources: grounding verifier tier when the grounded pipeline runs "
            "(default: config openbook.grounding_verifier_tier)"
        ),
    )
    ask.add_argument(
        "--triage-tier",
        help="triage tier to classify with (default: config triage_tier); without --chain or --stuffed it implies --route",
    )
    ask.add_argument("--effort", choices=["low", "medium", "high"], help="override the tier's effort")
    ask.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to models.yaml")
    ask.add_argument("--db", default=str(DEFAULT_DB_PATH), help="path to the telemetry SQLite file")
    ask.add_argument(
        "--simulate",
        choices=[
            "triage-timeout",
            "answer-error",
            "verifier-error",
            "draft-invalid",
            "quote-mismatch",
            "grounding-error",
        ],
        help=(
            "inject a failure to exercise the fail paths (verifier-error and draft-invalid need "
            "--chain; quote-mismatch and grounding-error need --sources without --stuffed and "
            "select the grounded pipeline; triage-timeout without --chain or --stuffed implies --route)"
        ),
    )
    ask.add_argument("--json", action="store_true", help="print the full run result as JSON")
    return parser


def format_trace(result: RunResult) -> str:
    lines = [f"trace_id      {result.trace_id}"]
    if result.triage_status == "ok":
        lines.append(
            f"triage        {result.triage_label} (confidence {result.triage_confidence:.2f}) - "
            f"{result.triage_reason}"
        )
    else:
        lines.append(f"triage        {result.triage_status}")
    if result.computation_verdict != "not_run":
        check_line = f"check         {result.computation_verdict} on {result.computation_check_tier}"
        if result.computation_check_confidence is not None:
            check_line += f" (confidence {result.computation_check_confidence:.2f})"
        if result.computation_check_reason:
            check_line += f" - {' '.join(result.computation_check_reason.split())}"
        lines.append(check_line)
    if result.auto_reason:
        lines.append(f"mode          {result.mode} ({result.auto_reason})")
    else:
        lines.append(f"mode          {result.mode}")
    lines.append(f"route         {result.selected_tier} (reason: {result.route_reason})")
    lines.append(
        f"final         {result.final_tier} / {result.final_model} "
        f"(effort: {result.effort}, fallback_used: {result.fallback_used})"
    )
    if result.fallback_skipped_reason:
        lines.append(f"fallback      skipped ({result.fallback_skipped_reason})")
    if result.mode == "chain":
        lines.append(
            f"chain         verifier {result.verifier_tier}, verdict {result.verification_status}, "
            f"repaired: {result.repaired}, schema_escalated: {result.schema_escalated}"
        )
        if result.verification_critique:
            lines.append(f"critique      {' '.join(result.verification_critique.split())}")
        if result.repair_error_type:
            lines.append(
                f"repair        failed ({result.repair_error_type}); the answer is the unrepaired draft"
            )
    if result.mode == "execute":
        if result.cross_check:
            lines.append(
                f"execution     {result.execution_status}, program A ({result.final_tier}) printed "
                f"{result.executed_answer!r}, program B ({result.cross_check_tier}) printed "
                f"{result.executed_answer_b!r}, compared as {result.cross_check_rule or 'nothing (not both ran)'}, "
                f"repaired: {result.repaired}"
            )
        else:
            lines.append(
                f"execution     {result.execution_status}, claimed "
                f"{'nothing (declined)' if result.claimed_answer is None else repr(result.claimed_answer)}, "
                f"executed {result.executed_answer!r}, repaired: {result.repaired}"
            )
        explanation = result.executed_explanation
        lines.append(
            "explanation   "
            + (f"{len(explanation)} characters printed by the program" if explanation else "none printed by the program")
        )
        for run in result.sandbox_runs:
            run_line = (
                f"sandbox {run.seq}     {(run.program_label + ' ' if result.cross_check else '') + run.attempt:<6} "
                f"{run.sandbox_name or '-'} {run.outcome}, "
                f"exit {run.exit_code}, run {run.duration_ms:.0f} ms, total {run.total_ms:.0f} ms, "
                f"network {run.network_policy}, env {','.join(run.environment_names)}, "
                f"removed: {run.sandbox_removed}"
            )
            if run.error_type:
                run_line += f"  [{run.error_type}]"
            lines.append(run_line)
        if result.repair_error_type:
            lines.append(f"repair        failed ({result.repair_error_type})")
    if result.mode == "route" and result.execution_status == "failed":
        lines.append("execution     failed; the answer is one closed-book call, NOT VERIFIED")
        if result.execution_note:
            lines.append(f"              {' '.join(result.execution_note.split())[:600]}")
    if result.mode in ("openbook", "stuffed"):
        lines.append(f"sources       {result.sources_count}")
    if result.mode == "openbook":
        retrieval_line = f"retrieval     {result.retrieval_method}, {result.passages_retrieved} passages"
        if result.retrieval_note:
            retrieval_line += f" ({result.retrieval_note})"
        lines.append(retrieval_line)
        lines.append(
            f"grounding     verifier {result.grounding_verifier_tier}, status {result.grounding_status}, "
            f"claims {result.claims_supported}/{result.claims_total} supported, "
            f"escalated: {result.repaired}, abstained: {result.abstained}, "
            f"schema_escalated: {result.schema_escalated}"
        )
        for note in result.grounding_notes:
            lines.append(f"              {' '.join(note.split())[:300]}")
        if result.escalation_passages is not None:
            lines.append(
                f"escalation    sent {result.escalation_passages} passages, "
                f"about {result.escalation_tokens_estimated} tokens"
            )
        if result.repair_error_type:
            lines.append(
                f"escalation    failed ({result.repair_error_type}); the result rests on the first draft"
            )
    for call in result.calls:
        call_line = (
            f"call {call.seq}        {call.role:<8} {call.tier:<11} {call.status:<7} "
            f"{call.latency_ms:>9.1f} ms"
        )
        if call.ollama_load_duration_ms is not None:
            call_line += f"  load {call.ollama_load_duration_ms:.1f} ms"
        if call.error_type:
            call_line += f"  [{call.error_type}]"
        if call.error_message:
            # Show why a call failed; without this a fallback hides the provider's reason.
            call_line += f"\n              {' '.join(call.error_message.split())[:300]}"
        lines.append(call_line)
    lines.append(f"total         {result.total_latency_ms:.1f} ms, ${result.total_cost_usd:.6f}")
    return "\n".join(lines)


def format_citations(result: RunResult) -> str:
    """Numbered citations with their quotes, printed under an open-book answer."""
    lines = ["", "Citations:"]
    for number, citation in enumerate(result.citations, start=1):
        quote = " ".join(citation.quote.split())
        lines.append(f'[{number}] {citation.source} ({citation.passage_id}): "{quote}"')
    return "\n".join(lines)


def _mode_from_arguments(arguments: argparse.Namespace) -> str:
    """Explicit flags win; anything else is auto."""
    if arguments.execute or arguments.cross_check or arguments.cross_check_tier:
        return "execute"
    if arguments.stuffed:
        return "stuffed"
    if arguments.chain:
        return "chain"
    if arguments.route:
        return "route"
    if arguments.sources:
        # These two failures exist only inside the grounded pipeline.
        if arguments.simulate in OPENBOOK_ONLY_SIMULATIONS:
            return "openbook"
        return "auto"
    # Both of these are about the triage call, which only route mode makes here.
    if arguments.triage_tier is not None or arguments.simulate == "triage-timeout":
        return "route"
    return "auto"


def _computation_check_from_arguments(arguments: argparse.Namespace) -> bool | None:
    """True or False when a flag was given, None to use auto.computation_check."""
    if arguments.computation_check:
        return True
    if arguments.no_computation_check:
        return False
    return None


def run_ask(arguments: argparse.Namespace) -> int:
    if arguments.chain and arguments.sources:
        print("error: --chain and --sources cannot be combined", file=sys.stderr)
        return EXIT_USAGE
    if arguments.route and (arguments.chain or arguments.stuffed or arguments.sources):
        print("error: --route cannot be combined with --chain, --stuffed or --sources", file=sys.stderr)
        return EXIT_USAGE
    if (arguments.execute or arguments.cross_check or arguments.cross_check_tier) and (
        arguments.chain or arguments.route or arguments.stuffed or arguments.sources
        or arguments.triage_tier or arguments.verifier_tier
    ):
        print(
            "error: --execute and --cross-check cannot be combined with --chain, --route, --stuffed, "
            "--sources, --triage-tier or --verifier-tier",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if arguments.computation_check and arguments.no_computation_check:
        print("error: --computation-check and --no-computation-check cannot be combined", file=sys.stderr)
        return EXIT_USAGE
    if arguments.require_citations and not arguments.sources:
        print(
            "error: --require-citations needs --sources: without sources there is nothing to cite",
            file=sys.stderr,
        )
        return EXIT_USAGE
    try:
        harness = build_harness(arguments.config, arguments.db, triage_tier=arguments.triage_tier)
        request = AskRequest.from_question(
            arguments.question,
            tier=arguments.tier,
            effort=arguments.effort,
            simulate=arguments.simulate,
            mode=_mode_from_arguments(arguments),
            verifier_tier=arguments.verifier_tier,
            sources=arguments.sources or [],
            require_citations=arguments.require_citations,
            cross_check=True if arguments.cross_check else None,
            cross_check_tier=arguments.cross_check_tier,
            computation_check=_computation_check_from_arguments(arguments),
        )
        result = harness.answer(request)
    except (
        ConfigError,
        UnknownTierError,
        ChainNotConfiguredError,
        OpenbookNotConfiguredError,
        ExecuteNotConfiguredError,
    ) as usage_error:
        print(f"error: {usage_error}", file=sys.stderr)
        return EXIT_USAGE
    except ValidationError as request_error:
        reasons = "; ".join(error["msg"] for error in request_error.errors())
        print(f"error: {reasons}", file=sys.stderr)
        return EXIT_USAGE

    if arguments.json:
        print(result.model_dump_json(indent=2))
    else:
        print(format_trace(result), file=sys.stderr)
        if result.status == "ok":
            print(result.answer)
            if result.citations:
                print(format_citations(result))
        else:
            print(
                f"error: run failed ({result.error_type}): {result.error_message}",
                file=sys.stderr,
            )

    if result.status == "ok":
        return EXIT_OK
    return EXIT_RUN_FAILED


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "ask":
        return run_ask(arguments)
    return EXIT_USAGE
