"""Eval runner tests: fake harness, fake judge, no network and no real database."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from inference_arbiter.grading import Grade

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_triage  # noqa: E402
import run_eval  # noqa: E402

CASES = [
    {
        "id": "LK-01",
        "question": "capital of Australia?",
        "expected_answer": "Canberra",
        "match": {"type": "contains", "any_of": ["Canberra"]},
        "category": "lookup",
        "expected_route": "local",
    },
    {
        "id": "CD-01",
        "question": "what does the query return?",
        "expected_answer": "2",
        "match": {"type": "numeric", "value": 2, "tolerance": 0},
        "category": "code",
        "expected_route": "code",
    },
    {
        "id": "RS-01",
        "question": "how far?",
        "expected_answer": "270",
        "match": {"type": "numeric", "value": 270, "tolerance": 0},
        "category": "reasoning",
        "expected_route": "mid",
    },
    {
        "id": "DM-01",
        "question": "what is ARQC?",
        "expected_answer": "Authorization Request Cryptogram",
        "match": {"type": "rubric", "criteria": ["Expands ARQC."], "pass_rule": "all criteria met"},
        "category": "domain",
        "expected_route": "mid",
    },
]
CORRECT_ANSWERS = {"LK-01": "Canberra", "CD-01": "2", "RS-01": "270", "DM-01": "good rubric answer"}
QUESTION_TO_ID = {case["question"]: case["id"] for case in CASES}
GOLDEN_INFO = {"path": "golden.yaml", "version": 1, "sha256": "golden-sha"}


def fake_result(
    answer: str | None,
    final_tier: str = "mid",
    *,
    status: str = "ok",
    fallback_used: bool = False,
    cost_usd: float = 0.0,
    latency_ms: float = 100.0,
    calls: list | None = None,
    **extra,
) -> SimpleNamespace:
    fields = dict(
        trace_id="trace",
        status=status,
        answer=answer,
        error_type=None if status == "ok" else "http_5xx",
        error_message=None if status == "ok" else "boom",
        triage_label=None,
        triage_confidence=None,
        route_reason="forced",
        selected_tier=final_tier,
        final_tier=final_tier,
        final_model=f"{final_tier}-model",
        fallback_used=fallback_used,
        total_latency_ms=latency_ms,
        total_cost_usd=cost_usd,
        calls=calls or [],
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


class FakeHarness:
    """Answers from a per-case table; records every request it receives."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return self._responder(QUESTION_TO_ID[request.messages[0].content], request)


class FakeJudge:
    """Passes an answer that contains 'good'; reports itself as the given model."""

    def __init__(self, model: str = "judge-model", cost_usd: float = 0.0) -> None:
        self.model = model
        self.cost_usd = cost_usd
        self.judged: list[str] = []

    def judge(self, case, answer) -> Grade:
        self.judged.append(case["id"])
        if "unjudgeable" in answer:
            return Grade(status="ungraded", reason="judge call failed", match_type="rubric",
                         judge_model=self.model, judge_cost_usd=self.cost_usd)
        status = "pass" if "good" in answer else "fail"
        return Grade(status=status, reason="judged", match_type="rubric", judge_tier="mid",
                     judge_model=self.model, judge_cost_usd=self.cost_usd)


def harness_answering(answers_by_tier: dict[str, dict[str, str]], **result_fields):
    """A factory whose harness answers each case from answers_by_tier[forced tier or 'router']."""
    harnesses: list[FakeHarness] = []

    def responder(case_id, request):
        tier = request.tier or "router"
        final_tier = request.tier or "mid"
        return fake_result(answers_by_tier[tier][case_id], final_tier, **result_fields)

    def factory(strategy):
        harness = FakeHarness(responder)
        harnesses.append(harness)
        return harness

    factory.harnesses = harnesses
    return factory


def record(strategy: str, case_id: str, outcome: str, *, category: str = "lookup",
           cost_usd: float = 0.0, latency_ms: float | None = 100.0, **extra) -> dict:
    fields = dict(strategy=strategy, case_id=case_id, category=category, outcome=outcome,
                  cost_usd=cost_usd, judge_cost_usd=0.0, latency_ms=latency_ms,
                  fallback_used=False, self_graded=False)
    fields.update(extra)
    return fields


# ---- strategies and requests ----


def test_parse_strategy():
    assert run_eval.parse_strategy("tier:mid") == run_eval.Strategy("tier:mid", "tier", "mid")
    assert run_eval.parse_strategy("router").kind == "router"
    assert run_eval.parse_strategy("chain").kind == "chain"
    for bad_text in ("tier:", "mid", "routerx"):
        with pytest.raises(ValueError):
            run_eval.parse_strategy(bad_text)


def test_build_request_per_strategy():
    tier_request = run_eval.build_request(run_eval.parse_strategy("tier:local"), "q", None)
    router_request = run_eval.build_request(run_eval.parse_strategy("router"), "q", None)

    assert tier_request.tier == "local"
    assert router_request.tier is None
    assert tier_request.messages[0].content == "q"
    if run_eval.chain_mode_supported():
        chain_request = run_eval.build_request(run_eval.parse_strategy("chain"), "q", None)
        assert tier_request.mode == "route"
        assert router_request.mode == "route"
        assert chain_request.mode == "chain"


def test_filter_cases_by_id_and_category():
    assert [case["id"] for case in run_eval.filter_cases(CASES, ["RS-01", "LK-01"], None)] == ["LK-01", "RS-01"]
    assert [case["id"] for case in run_eval.filter_cases(CASES, None, ["code", "domain"])] == ["CD-01", "DM-01"]
    assert run_eval.filter_cases(CASES, ["LK-01"], ["code"]) == []
    assert run_eval.filter_cases(CASES, None, None) == CASES


# ---- running and grading ----


def test_run_records_grades_for_every_match_type():
    factory = harness_answering({"mid": {**CORRECT_ANSWERS, "RS-01": "180"}}, cost_usd=0.002)

    run = run_eval.run_strategies(CASES, [run_eval.parse_strategy("tier:mid")], factory, FakeJudge())

    outcomes = {row["case_id"]: row["outcome"] for row in run["records"]}
    assert outcomes == {"LK-01": "pass", "CD-01": "pass", "RS-01": "fail", "DM-01": "pass"}
    reasoning_row = next(row for row in run["records"] if row["case_id"] == "RS-01")
    assert reasoning_row["answer"] == "180"
    assert reasoning_row["final_tier"] == "mid"
    assert reasoning_row["final_model"] == "mid-model"
    assert reasoning_row["cost_usd"] == 0.002
    assert "180" in reasoning_row["grade_reason"]
    assert run["cumulative_cost_usd"] == pytest.approx(0.008)
    assert run["stopped_early"] is False


def test_verification_fields_are_recorded_when_the_harness_returns_them():
    def responder(case_id, request):
        return fake_result(CORRECT_ANSWERS[case_id], "mid", mode="chain", verification_status="passed",
                           verification_critique="looks right", repaired=True, schema_escalated=False)

    harness = FakeHarness(responder)
    row = run_eval.run_case(harness, run_eval.parse_strategy("router"), CASES[0], FakeJudge())

    assert row["mode"] == "chain"
    assert row["verification_status"] == "passed"
    assert row["verification_critique"] == "looks right"
    assert row["repaired"] is True
    assert row["schema_escalated"] is False


def test_ungraded_rubric_is_neither_pass_nor_fail():
    factory = harness_answering({"mid": {**CORRECT_ANSWERS, "DM-01": "unjudgeable"}})

    run = run_eval.run_strategies(CASES, [run_eval.parse_strategy("tier:mid")], factory, FakeJudge())
    summary = run_eval.summarise_strategy("tier:mid", run["records"])

    assert summary["ungraded"] == 1
    assert summary["pass"] == 3
    assert summary["fail"] == 0
    assert summary["graded"] == 3
    assert summary["accuracy"] == 1.0


def test_tier_run_that_fell_back_is_flagged_and_excluded():
    failed_local_call = SimpleNamespace(tier="local", status="timeout", error_type="timeout")

    def responder(case_id, request):
        if case_id == "RS-01":
            # local failed and the router hopped to mid, which answered correctly.
            return fake_result("270", "mid", fallback_used=True, calls=[failed_local_call],
                               selected_tier="local")
        return fake_result(CORRECT_ANSWERS[case_id], "local")

    judge = FakeJudge()
    run = run_eval.run_strategies(
        CASES, [run_eval.parse_strategy("tier:local")], lambda strategy: FakeHarness(responder), judge
    )
    fallback_row = next(row for row in run["records"] if row["case_id"] == "RS-01")
    summary = run_eval.summarise_strategy("tier:local", run["records"])

    assert fallback_row["outcome"] == "fallback_excluded"
    assert fallback_row["fell_back_off_measured_tier"] is True
    assert fallback_row["measured_tier_error"] == "timeout"
    assert fallback_row["final_tier"] == "mid"
    assert fallback_row["grade"] is None
    assert summary["fallback_excluded"] == 1
    assert summary["pass"] == 3
    assert summary["graded"] == 3
    assert summary["accuracy"] == 1.0
    # The mid answer must not count as local's answer in the measured routes either.
    routes = run_eval.derive_measured_routes(CASES, run["records"])
    assert routes["RS-01"]["tier_outcomes"]["local"] == "fallback_excluded"
    assert routes["RS-01"]["route"] is None


def test_router_fallback_counts_because_fallback_is_part_of_the_strategy():
    def responder(case_id, request):
        return fake_result(CORRECT_ANSWERS[case_id], "mid", fallback_used=True)

    row = run_eval.run_case(FakeHarness(responder), run_eval.parse_strategy("router"), CASES[0], FakeJudge())

    assert row["outcome"] == "pass"
    assert row["fallback_used"] is True
    assert row["fell_back_off_measured_tier"] is False


def test_failed_run_and_raising_harness_are_errors_not_fails():
    def responder(case_id, request):
        if case_id == "LK-01":
            return fake_result(None, "mid", status="error")
        raise RuntimeError("harness exploded")

    run = run_eval.run_strategies(
        CASES[:2], [run_eval.parse_strategy("tier:mid")], lambda strategy: FakeHarness(responder), FakeJudge()
    )
    summary = run_eval.summarise_strategy("tier:mid", run["records"])

    assert [row["outcome"] for row in run["records"]] == ["error", "error"]
    assert run["records"][0]["error_type"] == "http_5xx"
    assert run["records"][1]["error_type"] == "exception"
    assert summary["error"] == 2
    assert summary["accuracy"] is None


def test_self_graded_flag_when_judge_model_wrote_the_answer():
    factory = harness_answering({"mid": CORRECT_ANSWERS, "local": CORRECT_ANSWERS})
    strategies = [run_eval.parse_strategy("tier:mid"), run_eval.parse_strategy("tier:local")]

    run = run_eval.run_strategies(CASES, strategies, factory, FakeJudge(model="mid-model"))
    flagged = [(row["strategy"], row["case_id"]) for row in run["records"] if row["self_graded"]]

    assert flagged == [("tier:mid", "DM-01")]
    assert run_eval.summarise_strategy("tier:mid", run["records"])["self_graded"] == 1


def test_cost_cap_stops_cleanly_after_the_cap_is_passed():
    factory = harness_answering({"mid": CORRECT_ANSWERS, "local": CORRECT_ANSWERS}, cost_usd=0.04)
    strategies = [run_eval.parse_strategy("tier:mid"), run_eval.parse_strategy("tier:local")]

    run = run_eval.run_strategies(CASES, strategies, factory, FakeJudge(), max_cost_usd=0.10)

    # 0.04, 0.08, 0.12: the third run passes the cap, so the fourth is never started.
    assert [row["case_id"] for row in run["records"]] == ["LK-01", "CD-01", "RS-01"]
    assert run["stopped_early"] is True
    assert "cost cap passed" in run["stop_reason"]
    assert "DM-01" in run["stop_reason"]
    assert run["cumulative_cost_usd"] == pytest.approx(0.12)
    assert len(factory.harnesses) == 1


def test_checkpoint_gets_the_run_so_far_after_every_case():
    factory = harness_answering({"mid": CORRECT_ANSWERS}, cost_usd=0.01)
    seen: list[tuple[list[str], float]] = []

    def checkpoint(run_state):
        seen.append(([row["case_id"] for row in run_state["records"]], run_state["cumulative_cost_usd"]))

    run_eval.run_strategies(
        CASES, [run_eval.parse_strategy("tier:mid")], factory, FakeJudge(), checkpoint=checkpoint
    )

    assert [case_ids for case_ids, _ in seen] == [
        ["LK-01"],
        ["LK-01", "CD-01"],
        ["LK-01", "CD-01", "RS-01"],
        ["LK-01", "CD-01", "RS-01", "DM-01"],
    ]
    assert seen[0][1] == pytest.approx(0.01)


def test_cost_cap_counts_judge_cost():
    factory = harness_answering({"mid": {**CORRECT_ANSWERS}})
    rubric_first = [CASES[3], CASES[0], CASES[1]]

    run = run_eval.run_strategies(
        rubric_first, [run_eval.parse_strategy("tier:mid")], factory, FakeJudge(cost_usd=0.5), max_cost_usd=0.10
    )

    assert [row["case_id"] for row in run["records"]] == ["DM-01"]
    assert run["stopped_early"] is True


def test_no_cost_cap_runs_everything():
    factory = harness_answering({"mid": CORRECT_ANSWERS}, cost_usd=100.0)

    run = run_eval.run_strategies(CASES, [run_eval.parse_strategy("tier:mid")], factory, FakeJudge())

    assert len(run["records"]) == len(CASES)
    assert run["stopped_early"] is False


def test_chain_reported_not_runnable_when_mode_is_missing(monkeypatch):
    monkeypatch.setattr(run_eval, "chain_mode_supported", lambda: False)
    factory = harness_answering({"mid": CORRECT_ANSWERS})
    strategies = [run_eval.parse_strategy("chain"), run_eval.parse_strategy("tier:mid")]

    run = run_eval.run_strategies(CASES, strategies, factory, FakeJudge())

    assert "chain" in run["not_runnable"]
    assert {row["strategy"] for row in run["records"]} == {"tier:mid"}


def test_strategy_whose_harness_cannot_be_built_is_not_runnable():
    def factory(strategy):
        raise RuntimeError("no chain config")

    run = run_eval.run_strategies(CASES, [run_eval.parse_strategy("router")], factory, FakeJudge())

    assert run["records"] == []
    assert "no chain config" in run["not_runnable"]["router"]


# ---- scorecard arithmetic ----


def test_percentile_is_nearest_rank():
    values = [float(number) for number in range(1, 21)]

    assert run_eval.percentile(values, 50) == 10.0
    assert run_eval.percentile(values, 95) == 19.0
    assert run_eval.percentile([5.0], 95) == 5.0
    assert run_eval.percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert run_eval.percentile([], 50) is None


def test_wilson_interval():
    low, high = run_eval.wilson_interval(18, 25)

    assert low == pytest.approx(0.5242, abs=0.001)
    assert high == pytest.approx(0.8572, abs=0.001)
    assert run_eval.wilson_interval(0, 0) is None
    assert run_eval.wilson_interval(5, 5)[1] == 1.0


def test_summarise_strategy_arithmetic():
    records = [
        record("tier:mid", "A", "pass", category="lookup", cost_usd=0.01, latency_ms=100.0),
        record("tier:mid", "B", "pass", category="lookup", cost_usd=0.02, latency_ms=300.0),
        record("tier:mid", "C", "fail", category="code", cost_usd=0.03, latency_ms=200.0),
        record("tier:mid", "D", "ungraded", category="domain", cost_usd=0.04, latency_ms=400.0),
        record("tier:mid", "E", "error", category="domain", cost_usd=0.0, latency_ms=9000.0),
        record("tier:mid", "F", "fallback_excluded", category="code", cost_usd=0.05, latency_ms=8000.0,
               fallback_used=True),
        record("tier:local", "A", "fail", category="lookup"),
    ]

    summary = run_eval.summarise_strategy("tier:mid", records)

    assert summary["cases"] == 6
    assert (summary["pass"], summary["fail"], summary["ungraded"]) == (2, 1, 1)
    assert (summary["error"], summary["fallback_excluded"]) == (1, 1)
    assert summary["graded"] == 3
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["total_cost_usd"] == pytest.approx(0.15)
    assert summary["cost_per_question_usd"] == pytest.approx(0.025)
    # Latency covers answered runs only: error and fallback runs are left out.
    assert summary["latency_p50_ms"] == 200.0
    assert summary["latency_p95_ms"] == 400.0
    assert summary["fallback_used"] == 1
    assert summary["by_category"]["lookup"] == {"cases": 2, "pass": 2, "graded": 2, "accuracy": 1.0}
    assert summary["by_category"]["code"] == {"cases": 2, "pass": 0, "graded": 1, "accuracy": 0.0}
    assert summary["by_category"]["domain"] == {"cases": 2, "pass": 0, "graded": 0, "accuracy": None}


def test_head_to_head_counts_and_cost_ratio():
    records = [
        record("tier:mid", "A", "pass", cost_usd=0.01),
        record("tier:mid", "B", "fail", cost_usd=0.01),
        record("tier:mid", "C", "pass", cost_usd=0.01),
        record("tier:mid", "D", "fail", cost_usd=0.01),
        record("tier:mid", "E", "ungraded", cost_usd=0.01),
        record("tier:mid", "F", "pass", cost_usd=0.01),
        record("chain", "A", "pass", cost_usd=0.03),
        record("chain", "B", "pass", cost_usd=0.03),
        record("chain", "C", "fail", cost_usd=0.03),
        record("chain", "D", "fail", cost_usd=0.03),
        record("chain", "E", "pass", cost_usd=0.03),
        # F was not run by chain: not comparable and outside the cost ratio.
    ]

    comparison = run_eval.head_to_head("chain", "tier:mid", records)

    assert comparison["both_right"] == 1
    assert comparison["won"] == 1
    assert comparison["lost"] == 1
    assert comparison["both_wrong"] == 1
    assert comparison["not_comparable"] == 2
    assert comparison["won_cases"] == ["B"]
    assert comparison["lost_cases"] == ["C"]
    assert comparison["cost_ratio"] == pytest.approx(3.0)


def test_head_to_head_cost_ratio_is_none_for_a_free_baseline():
    records = [record("tier:mid", "A", "pass"), record("chain", "A", "pass", cost_usd=0.03)]

    assert run_eval.head_to_head("chain", "tier:mid", records)["cost_ratio"] is None


def test_build_scorecard_compares_every_strategy_with_the_baseline():
    records = [
        record("tier:local", "A", "fail"),
        record("tier:mid", "A", "pass"),
        record("router", "A", "pass"),
    ]

    scorecard = run_eval.build_scorecard(records)

    assert [summary["strategy"] for summary in scorecard["strategies"]] == ["tier:local", "tier:mid", "router"]
    assert [comparison["strategy"] for comparison in scorecard["head_to_head"]] == ["tier:local", "router"]
    assert scorecard["baseline_present"] is True
    assert run_eval.build_scorecard([record("router", "A", "pass")])["head_to_head"] == []


def test_print_scorecard_states_the_small_sample_caveat(capsys):
    records = [record("tier:mid", "A", "pass"), record("router", "A", "fail")]

    run_eval.print_scorecard(run_eval.build_scorecard(records), case_count=1)
    printed = capsys.readouterr().out

    assert "small sample" in printed
    assert "Head-to-head against tier:mid" in printed
    assert "n/a (baseline cost 0)" in printed


# ---- measured routes ----


def tier_records(outcomes_by_tier: dict[str, dict[str, str]]) -> list[dict]:
    return [
        record(f"tier:{tier_name}", case_id, outcome)
        for tier_name, outcomes in outcomes_by_tier.items()
        for case_id, outcome in outcomes.items()
    ]


def test_measured_route_is_the_cheapest_correct_tier():
    records = tier_records(
        {
            "local": {"LK-01": "pass", "CD-01": "fail", "RS-01": "fail", "DM-01": "fail"},
            "code": {"LK-01": "pass", "CD-01": "pass"},
            "mid": {"LK-01": "pass", "CD-01": "pass", "RS-01": "pass", "DM-01": "fail"},
            "frontier": {"LK-01": "pass", "CD-01": "pass", "RS-01": "pass", "DM-01": "fail"},
        }
    )

    routes = run_eval.derive_measured_routes(CASES, records)

    assert (routes["LK-01"]["status"], routes["LK-01"]["route"]) == ("measured", "local")
    # code is not considered outside the code category, even though it passed.
    assert routes["LK-01"]["acceptable"] == ["local"]
    assert "code" not in routes["LK-01"]["tier_outcomes"]
    assert (routes["CD-01"]["status"], routes["CD-01"]["route"]) == ("measured", "code")
    assert (routes["RS-01"]["status"], routes["RS-01"]["route"]) == ("measured", "mid")
    assert routes["RS-01"]["missing_tiers"] == []
    assert (routes["DM-01"]["status"], routes["DM-01"]["route"]) == ("none_correct", None)


def test_measured_route_lists_both_local_tiers_when_both_answer_a_code_case():
    records = tier_records({"local": {"CD-01": "pass"}, "code": {"CD-01": "pass"}, "mid": {"CD-01": "pass"}})

    route = run_eval.derive_measured_routes(CASES, records)["CD-01"]

    assert route["route"] == "code"
    assert route["acceptable"] == ["code", "local"]
    assert route["status"] == "measured"
    assert route["missing_tiers"] == ["frontier"]


def test_measured_route_marks_missing_tiers():
    records = tier_records(
        {
            "local": {"LK-01": "ungraded", "RS-01": "fail", "DM-01": "fail"},
            "mid": {"LK-01": "pass", "RS-01": "fail", "DM-01": "error"},
        }
    )

    routes = run_eval.derive_measured_routes(CASES, records)

    # mid is right but the cheaper local tier has no pass/fail, so mid is only provisional.
    assert (routes["LK-01"]["status"], routes["LK-01"]["route"]) == ("provisional", "mid")
    assert routes["LK-01"]["missing_tiers"] == ["local", "frontier"]
    # Nothing right yet and frontier never ran: unresolved, not none_correct.
    assert (routes["RS-01"]["status"], routes["RS-01"]["route"]) == ("unresolved", None)
    assert routes["RS-01"]["missing_tiers"] == ["frontier"]
    assert routes["DM-01"]["missing_tiers"] == ["mid", "frontier"]
    assert routes["CD-01"]["status"] == "unresolved"
    assert routes["CD-01"]["missing_tiers"] == ["code", "local", "mid", "frontier"]


def test_measured_routes_ignore_router_and_chain_results():
    records = [record("router", "LK-01", "pass"), record("chain", "LK-01", "pass")]

    assert run_eval.derive_measured_routes(CASES, records)["LK-01"]["status"] == "unresolved"


def test_summarise_and_write_measured_routes(tmp_path):
    records = tier_records(
        {
            "local": {"LK-01": "pass", "RS-01": "fail", "DM-01": "fail"},
            "mid": {"LK-01": "pass", "RS-01": "fail", "DM-01": "pass"},
            "frontier": {"RS-01": "fail"},
        }
    )
    routes = run_eval.derive_measured_routes(CASES, records)
    routes_path = tmp_path / "eval" / "measured_routes.json"

    run_eval.write_measured_routes(routes_path, routes, GOLDEN_INFO, ["run.json"])
    written = json.loads(routes_path.read_text(encoding="utf-8"))

    assert written["summary"]["by_route"] == {"local": 1, "mid": 1}
    assert written["summary"]["by_status"]["none_correct"] == ["RS-01"]
    assert written["summary"]["by_status"]["unresolved"] == ["CD-01"]
    assert written["summary"]["resolved"] == 2
    assert written["summary"]["hand_written_route_agrees"] == 2
    assert written["routes"]["DM-01"]["route"] == "mid"
    assert "code" in written["ordering"]
    assert written["golden"]["sha256"] == "golden-sha"


# ---- merging earlier results ----


def test_merge_adds_only_pairs_this_run_did_not_produce(tmp_path):
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "golden": GOLDEN_INFO,
                "records": [record("tier:mid", "A", "fail"), record("tier:local", "A", "pass")],
            }
        ),
        encoding="utf-8",
    )
    new_records = [record("tier:mid", "A", "pass"), record("chain", "A", "pass")]

    merged, used, warnings = run_eval.merge_prior_records(new_records, [str(prior_path)], GOLDEN_INFO)

    by_key = {(row["strategy"], row["case_id"]): row for row in merged}
    assert len(merged) == 3
    assert by_key[("tier:mid", "A")]["outcome"] == "pass"
    assert by_key[("tier:local", "A")]["merged_from"] == str(prior_path)
    assert used == [str(prior_path)]
    assert warnings == []


def test_merge_skips_a_file_graded_against_another_golden_set(tmp_path):
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps({"golden": {"sha256": "other"}, "records": [record("tier:local", "A", "pass")]}),
        encoding="utf-8",
    )

    merged, used, warnings = run_eval.merge_prior_records(
        [], [str(prior_path), str(tmp_path / "absent.json")], GOLDEN_INFO
    )

    assert merged == []
    assert used == []
    assert len(warnings) == 2
    assert "different golden set" in warnings[0]


# ---- compare_triage against measured routes ----


def measured_entry(status: str, route: str | None, acceptable: list[str]) -> dict:
    return {"status": status, "route": route, "acceptable": acceptable}


def test_score_against_measured_correct_under_and_over():
    measured = measured_entry("measured", "mid", ["mid"])

    assert compare_triage.score_against_measured("mid", measured)["measured_correct"] is True
    under = compare_triage.score_against_measured("local", measured)
    over = compare_triage.score_against_measured("frontier", measured)
    assert (under["measured_correct"], under["measured_under"], under["measured_over"]) == (False, True, False)
    assert (over["measured_correct"], over["measured_under"], over["measured_over"]) == (False, False, True)


def test_score_against_measured_accepts_either_local_tier_when_both_were_right():
    measured = measured_entry("measured", "code", ["code", "local"])

    assert compare_triage.score_against_measured("local", measured)["measured_correct"] is True
    assert compare_triage.score_against_measured("code", measured)["measured_correct"] is True
    # Same strength but not a tier that answered it: wrong, yet neither under nor over.
    only_code = compare_triage.score_against_measured("local", measured_entry("measured", "code", ["code"]))
    assert (only_code["measured_correct"], only_code["measured_under"], only_code["measured_over"]) == (
        False, False, False,
    )


@pytest.mark.parametrize("measured", [None, measured_entry("unresolved", None, []),
                                      measured_entry("none_correct", None, [])])
def test_score_against_measured_leaves_unresolved_cases_unscored(measured):
    scored = compare_triage.score_against_measured("mid", measured)

    assert scored["measured_correct"] is None
    assert scored["measured_under"] is None


def test_compare_triage_summary_reports_both_scores():
    def row(case_id, routed_correct, measured_correct, measured_status="measured", under=False, over=False):
        scorable = measured_correct is not None
        return {
            "candidate": "triage", "case_id": case_id, "status": "ok", "latency_ms": 10.0, "cost_usd": 0.0,
            "label_correct": routed_correct, "routed_correct": routed_correct,
            "under_routed": False, "over_routed": not routed_correct,
            "measured_status": measured_status, "measured_correct": measured_correct,
            "measured_under": under if scorable else None, "measured_over": over if scorable else None,
        }

    rows = [
        row("A", True, True),
        row("B", False, True, measured_status="provisional"),
        row("C", True, False, under=True),
        row("D", True, None, measured_status="unresolved"),
    ]

    summary = compare_triage.summarise(rows, "triage")

    assert summary["routed_correct"] == 3
    assert summary["measured_scored"] == 3
    assert summary["measured_unscored"] == 1
    assert summary["measured_provisional"] == 1
    assert summary["measured_correct"] == 2
    assert summary["measured_under"] == 1
    assert summary["measured_over"] == 0


def test_compare_triage_summary_without_routes_has_no_measured_fields():
    rows = [{"candidate": "triage", "status": "ok", "latency_ms": 10.0, "cost_usd": 0.0,
             "label_correct": True, "routed_correct": True, "under_routed": False, "over_routed": False}]

    assert "measured_scored" not in compare_triage.summarise(rows, "triage")


def test_load_measured_routes_round_trips_the_file_run_eval_writes(tmp_path):
    records = tier_records({"local": {"LK-01": "pass"}})
    routes_path = tmp_path / "measured_routes.json"
    run_eval.write_measured_routes(
        routes_path, run_eval.derive_measured_routes(CASES, records), GOLDEN_INFO, []
    )

    loaded = compare_triage.load_measured_routes(routes_path)

    assert compare_triage.score_against_measured("local", loaded["LK-01"])["measured_correct"] is True
    assert compare_triage.score_against_measured("mid", loaded["LK-01"])["measured_over"] is True


# ---- regrading stored answers ----


def test_regrade_grades_stored_answers_again_without_running_a_harness():
    factory = harness_answering({"mid": {**CORRECT_ANSWERS, "RS-01": "180", "DM-01": "poor answer"}})
    run = run_eval.run_strategies(CASES, [run_eval.parse_strategy("tier:mid")], factory, FakeJudge())
    stored = json.loads(json.dumps(run["records"]))
    # Simulate a stale grade from an older grader, plus records that have nothing to grade.
    stored[0]["outcome"] = stored[0]["grade"] = "fail"
    stored.append(record("tier:mid", "LK-01", "error", answer=None, final_model="mid-model"))
    stored.append(record("tier:local", "RS-01", "fallback_excluded", answer="270", final_model="mid-model"))

    regraded = run_eval.regrade_records(stored, CASES, FakeJudge(model="mid-model"))

    assert [row["outcome"] for row in regraded] == [
        "pass", "pass", "fail", "fail", "error", "fallback_excluded",
    ]
    assert regraded[3]["self_graded"] is True
    assert stored[0]["outcome"] == "fail"
    assert len(factory.harnesses) == 1


def test_summary_counts_adjudicated_answers():
    records = [
        record("tier:mid", "A", "pass", grade_detail={"adjudicated": True}),
        record("tier:mid", "B", "pass", grade_detail={}),
        record("tier:mid", "C", "error", grade_detail=None),
    ]

    assert run_eval.summarise_strategy("tier:mid", records)["adjudicated"] == 1


# ---- measured routes are only written from results that cover the tiers they need ----


def full_tier_records() -> list[dict]:
    return tier_records(
        {
            "local": {"LK-01": "pass", "CD-01": "fail", "RS-01": "fail", "DM-01": "fail"},
            "mid": {"LK-01": "pass", "CD-01": "pass", "RS-01": "pass", "DM-01": "fail"},
            "frontier": {"LK-01": "pass", "CD-01": "pass", "RS-01": "pass", "DM-01": "pass"},
            "code": {"CD-01": "pass"},
        }
    )


def test_route_coverage_has_no_gaps_when_every_needed_tier_ran_every_case():
    assert run_eval.route_coverage_gaps(CASES, full_tier_records()) == []


def test_route_coverage_names_a_tier_that_did_not_run():
    records = [row for row in full_tier_records() if row["strategy"] != "tier:frontier"]
    assert run_eval.route_coverage_gaps(CASES, records) == ["tier:frontier has no results"]
    only_router = [record("router", "LK-01", "pass")]
    assert run_eval.route_coverage_gaps(CASES, only_router) == [
        "tier:local has no results", "tier:mid has no results", "tier:frontier has no results",
        "tier:code has no results",
    ]


def test_route_coverage_names_a_tier_run_on_only_some_cases():
    records = [
        row for row in full_tier_records() if not (row["strategy"] == "tier:mid" and row["case_id"] != "LK-01")
    ]
    assert run_eval.route_coverage_gaps(CASES, records) == ["tier:mid has no result for 3 of 4 cases"]


def test_route_coverage_rejects_a_tier_whose_every_run_errored_or_fell_back():
    records = [row for row in full_tier_records() if row["strategy"] != "tier:frontier"]
    records += [record("tier:frontier", case["id"], "fallback_excluded") for case in CASES]
    assert run_eval.route_coverage_gaps(CASES, records) == [
        "tier:frontier has no pass or fail result (every run errored or fell back)"
    ]


def test_route_coverage_needs_the_code_tier_only_on_code_cases():
    without_code_category = [case for case in CASES if case["category"] != "code"]
    records = [row for row in full_tier_records() if row["strategy"] != "tier:code" and row["case_id"] != "CD-01"]
    assert run_eval.route_coverage_gaps(without_code_category, records) == []
    assert run_eval.route_coverage_gaps(CASES, [row for row in full_tier_records() if row["strategy"] != "tier:code"]) == [
        "tier:code has no results"
    ]


@pytest.fixture
def closed_book_main(monkeypatch, tmp_path):
    """main() over a two-case golden file with a fake harness; returns (argument builder, routes path)."""
    import yaml

    from conftest import REAL_CONFIG_PATH

    golden_cases = [CASES[0], CASES[2]]
    golden_path = tmp_path / "golden.yaml"
    golden_path.write_text(yaml.safe_dump({"version": 1, "cases": golden_cases}), encoding="utf-8")
    routes_path = tmp_path / "measured_routes.json"
    routes_path.write_text("EXISTING ROUTES", encoding="utf-8")

    def responder(case_id, request):
        return fake_result(CORRECT_ANSWERS[case_id], request.tier)

    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setattr(run_eval, "build_clients", lambda config: {"ollama": object(), "openrouter": object()})
    monkeypatch.setattr(run_eval, "build_harness", lambda **arguments: FakeHarness(responder))

    def arguments(*strategies, extra=()):
        return [
            "--strategies", *strategies, "--config", str(REAL_CONFIG_PATH), "--golden", str(golden_path),
            "--out", str(tmp_path / "out.json"), "--db", str(tmp_path / "t.db"),
            "--routes-out", str(routes_path), *extra,
        ]

    return arguments, routes_path


class ProcessKilled(BaseException):
    """Stands in for a kill: nothing in the runner catches it."""


def test_main_keeps_the_finished_cases_of_a_run_that_is_killed(closed_book_main, monkeypatch, tmp_path):
    arguments, _ = closed_book_main
    out_path = tmp_path / "out.json"

    def responder(case_id, request):
        if case_id == "RS-01":
            raise ProcessKilled()
        return fake_result(CORRECT_ANSWERS[case_id], request.tier)

    monkeypatch.setattr(run_eval, "build_harness", lambda **harness_arguments: FakeHarness(responder))

    with pytest.raises(ProcessKilled):
        run_eval.main(arguments("tier:mid"))

    partial = json.loads(out_path.read_text(encoding="utf-8"))
    assert partial["complete"] is False
    assert partial["scorecard"] is None
    assert [row["case_id"] for row in partial["records"]] == ["LK-01"]
    assert partial["records"][0]["outcome"] == "pass"
    assert not (tmp_path / "out.json.tmp").exists()


def test_main_regrades_the_file_a_killed_run_left(closed_book_main, monkeypatch, tmp_path):
    arguments, _ = closed_book_main
    partial_path = tmp_path / "out.json"
    regraded_path = tmp_path / "regraded.json"

    def responder(case_id, request):
        if case_id == "RS-01":
            raise ProcessKilled()
        return fake_result(CORRECT_ANSWERS[case_id], request.tier)

    monkeypatch.setattr(run_eval, "build_harness", lambda **harness_arguments: FakeHarness(responder))
    with pytest.raises(ProcessKilled):
        run_eval.main(arguments("tier:mid"))

    regrade_arguments = arguments("tier:mid", extra=("--regrade", str(partial_path)))
    regrade_arguments[regrade_arguments.index("--out") + 1] = str(regraded_path)
    assert run_eval.main(regrade_arguments) == 0

    regraded = json.loads(regraded_path.read_text(encoding="utf-8"))
    assert regraded["complete"] is True
    assert [row["case_id"] for row in regraded["records"]] == ["LK-01"]
    assert regraded["scorecard"]["strategies"][0]["cases"] == 1


def test_main_marks_a_finished_run_complete(closed_book_main, tmp_path):
    arguments, _ = closed_book_main

    assert run_eval.main(arguments("tier:mid")) == 0

    result = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert result["scorecard"] is not None
    assert [row["case_id"] for row in result["records"]] == ["LK-01", "RS-01"]


def test_main_does_not_overwrite_measured_routes_from_a_run_that_lacks_tiers(closed_book_main, capsys):
    arguments, routes_path = closed_book_main

    assert run_eval.main(arguments("tier:mid")) == 0

    printed = capsys.readouterr().out
    assert routes_path.read_text(encoding="utf-8") == "EXISTING ROUTES"
    assert "Measured routes not written: tier:local has no results; tier:frontier has no results" in printed
    assert "NOT written" in printed


def test_main_leaves_measured_routes_alone_when_no_tier_strategy_ran(closed_book_main, capsys):
    arguments, routes_path = closed_book_main
    assert run_eval.main(arguments("router")) == 0
    assert routes_path.read_text(encoding="utf-8") == "EXISTING ROUTES"


def test_main_writes_measured_routes_when_every_needed_tier_ran(closed_book_main, capsys):
    arguments, routes_path = closed_book_main

    assert run_eval.main(arguments("tier:local", "tier:mid", "tier:frontier")) == 0

    written = json.loads(routes_path.read_text(encoding="utf-8"))
    assert written["routes"]["LK-01"]["route"] == "local"
    assert written["summary"]["by_status"] == {"measured": ["LK-01", "RS-01"]}


def test_main_writes_partial_measured_routes_only_when_asked(closed_book_main, capsys):
    arguments, routes_path = closed_book_main

    assert run_eval.main(arguments("tier:mid", extra=("--allow-partial-routes",))) == 0

    written = json.loads(routes_path.read_text(encoding="utf-8"))
    assert written["routes"]["LK-01"]["status"] == "provisional"


def test_main_counts_merged_results_towards_route_coverage(closed_book_main, capsys, tmp_path):
    arguments, routes_path = closed_book_main
    assert run_eval.main(arguments("tier:local", "tier:frontier")) == 0
    assert routes_path.read_text(encoding="utf-8") == "EXISTING ROUTES"
    earlier = tmp_path / "earlier.json"
    earlier.write_text((tmp_path / "out.json").read_text(encoding="utf-8"), encoding="utf-8")

    assert run_eval.main(arguments("tier:mid", extra=("--merge", str(earlier)))) == 0

    assert json.loads(routes_path.read_text(encoding="utf-8"))["summary"]["resolved"] == 2


def test_head_to_head_reports_right_answers_where_the_baseline_errored_without_calling_them_wins():
    records = [
        record("openbook", "A", "pass"), record("stuffed:mid", "A", "error"),
        record("openbook", "B", "fail"), record("stuffed:mid", "B", "error"),
        record("openbook", "C", "pass"), record("stuffed:mid", "C", "fail"),
    ]
    comparison = run_eval.head_to_head("openbook", "stuffed:mid", records)
    assert comparison["won"] == 1
    assert comparison["won_cases"] == ["C"]
    assert comparison["not_comparable"] == 2
    assert comparison["right_where_baseline_errored"] == 1


# ---- escalations: attempted, succeeded and failed are counted apart ----


def escalation_record(case_id, **fields):
    record = {"strategy": "openbook", "case_id": case_id, "repaired": False, "repair_error_type": None,
              "escalation_attempted": False, "escalation_cost_usd": 0.0, "trace_id": f"trace-{case_id}"}
    record.update(fields)
    return record


def test_summarise_escalations_separates_failed_from_succeeded():
    records = [
        escalation_record("A1"),
        escalation_record("A2", repaired=True, escalation_attempted=True, escalation_cost_usd=0.18),
        escalation_record("A3", repair_error_type="empty", escalation_attempted=True, escalation_cost_usd=0.2),
        escalation_record("A4", repair_error_type="schema_invalid", escalation_attempted=True,
                          escalation_cost_usd=0.08),
    ]

    summary = run_eval.summarise_escalations(records)

    assert summary["escalations_attempted"] == 3
    assert summary["escalations_succeeded"] == 1
    assert summary["escalations_failed"] == 2
    assert summary["escalation_cost_usd"] == 0.46
    assert summary["failed_escalation_cost_usd"] == 0.28
    assert summary["failed_escalation_cases"] == ["A3", "A4"]


def test_summarise_escalations_reads_older_records_without_the_attempted_field():
    older_failed = {"strategy": "openbook", "case_id": "B1", "repaired": False, "repair_error_type": "empty"}
    older_plain = {"strategy": "openbook", "case_id": "B2", "repaired": False, "repair_error_type": None}

    summary = run_eval.summarise_escalations([older_failed, older_plain])

    assert (summary["escalations_attempted"], summary["escalations_failed"]) == (1, 1)
    # No per-call cost on the record: unknown, not zero.
    assert summary["failed_escalation_cost_usd"] is None


def test_backfill_escalation_costs_reads_the_telemetry_calls(tmp_path):
    import sqlite3

    db_path = tmp_path / "telemetry.db"
    connection = sqlite3.connect(str(db_path))
    connection.execute("CREATE TABLE runs (trace_id TEXT PRIMARY KEY)")
    connection.execute("CREATE TABLE calls (trace_id TEXT, role TEXT, cost_usd REAL)")
    connection.executemany("INSERT INTO runs VALUES (?)", [("t-failed",), ("t-plain",)])
    connection.executemany(
        "INSERT INTO calls VALUES (?, ?, ?)",
        [("t-failed", "draft", 0.0), ("t-failed", "ground", 0.0005), ("t-failed", "escalate", 0.196),
         ("t-plain", "draft", 0.0)],
    )
    connection.commit()
    connection.close()
    failed = {"case_id": "C1", "trace_id": "t-failed", "repaired": False, "repair_error_type": "empty"}
    plain = {"case_id": "C2", "trace_id": "t-plain", "repaired": False, "repair_error_type": None}
    unknown = {"case_id": "C3", "trace_id": "t-not-in-db", "repaired": False, "repair_error_type": "empty"}
    already = {"case_id": "C4", "trace_id": "t-failed", "escalation_cost_usd": 0.5}

    filled = run_eval.backfill_escalation_costs([failed, plain, unknown, already], db_path)

    assert filled == 2
    assert (failed["escalation_attempted"], failed["escalation_cost_usd"]) == (True, 0.196)
    assert (plain["escalation_attempted"], plain["escalation_cost_usd"]) == (False, 0.0)
    assert "escalation_cost_usd" not in unknown
    assert already["escalation_cost_usd"] == 0.5


def test_backfill_escalation_costs_without_a_database_changes_nothing(tmp_path):
    record = {"case_id": "C1", "trace_id": "t", "repaired": False}

    assert run_eval.backfill_escalation_costs([record], tmp_path / "missing.db") == 0
    assert "escalation_cost_usd" not in record


def test_copy_result_fields_records_the_escalation_call_cost():
    from types import SimpleNamespace

    calls = [
        SimpleNamespace(role="draft", tier="mid", cost_usd=0.0, error_message=None),
        SimpleNamespace(role="escalate", tier="frontier", cost_usd=0.079, error_message="invalid JSON"),
    ]
    result = SimpleNamespace(
        answer="kept draft", selected_tier="mid", final_tier="mid", final_model="m", fallback_used=False,
        triage_label=None, triage_confidence=None, route_reason="configured", calls=calls,
        total_latency_ms=10.0, total_cost_usd=0.079, error_type=None, error_message=None, trace_id="t",
        repaired=False, repair_error_type="schema_invalid",
    )
    record = {}

    run_eval.copy_result_fields(record, result)

    assert record["escalation_attempted"] is True
    assert record["escalation_cost_usd"] == 0.079
    assert run_eval.escalation_outcome(record) == "failed"


# ---- execute:<tier> strategy ----


def test_parse_execute_strategy():
    assert run_eval.parse_strategy("execute:mid") == run_eval.Strategy("execute:mid", "execute", "mid")
    assert "execute" in run_eval.SUITE_KINDS[run_eval.SUITE_CLOSEDBOOK]
    assert "execute" not in run_eval.SUITE_KINDS[run_eval.SUITE_OPENBOOK]
    for bad_text in ("execute", "execute:"):
        with pytest.raises(ValueError):
            run_eval.parse_strategy(bad_text)


def test_execute_strategy_requests_execute_mode_on_its_tier():
    request = run_eval.build_request(run_eval.parse_strategy("execute:mid"), "q", None)

    assert (request.mode, request.tier) == ("execute", "mid")
    assert run_eval.unsupported_reason(run_eval.parse_strategy("execute:mid")) is None


def test_execution_fields_are_recorded_per_case():
    class ExecutingHarness:
        def answer(self, request):
            return fake_result(
                "Corrected by execution: ...\n\nAnswer: 2", mode="execute", execution_status="corrected",
                claimed_answer="3", executed_answer="2", repaired=False,
            )

    case = next(case for case in CASES if case["id"] == "CD-01")
    row = run_eval.run_case(ExecutingHarness(), run_eval.parse_strategy("execute:mid"), case, None)

    assert row["outcome"] == "pass"
    assert (row["mode"], row["execution_status"], row["claimed_answer"], row["executed_answer"]) == (
        "execute", "corrected", "3", "2"
    )


def execute_record(case_id: str, outcome: str, execution_status: str | None, **extra) -> dict:
    return record("execute:mid", case_id, outcome, mode="execute", execution_status=execution_status, **extra)


def test_summarise_execution_counts_each_status_with_its_pass_rate():
    records = [
        execute_record("A", "pass", "claim_matched"),
        execute_record("B", "pass", "claim_matched", repaired=True),
        execute_record("C", "fail", "claim_matched"),
        execute_record("D", "pass", "corrected"),
        execute_record("E", "fail", "unverified"),
        execute_record("F", "ungraded", "unverified"),
        execute_record("G", "error", "skipped"),
        execute_record("H", "pass", "executed"),
        execute_record("I", "fail", "executed"),
        execute_record("J", "error", "failed"),
        record("tier:mid", "A", "pass", mode="route", execution_status="skipped"),
    ]

    execution = run_eval.summarise_strategy("execute:mid", records)["execution"]

    assert execution["claim_matched"] == {"count": 3, "pass": 2, "graded": 3}
    assert execution["verified"] == {"count": 0, "pass": 0, "graded": 0}
    assert execution["corrected"] == {"count": 1, "pass": 1, "graded": 1}
    assert execution["unverified"] == {"count": 2, "pass": 0, "graded": 1}
    assert execution["executed"] == {"count": 2, "pass": 1, "graded": 2}
    assert execution["failed"] == {"count": 1, "pass": 0, "graded": 0}
    assert execution["repaired"] == 1
    assert run_eval.summarise_strategy("tier:mid", records)["execution"] is None


def test_scorecard_prints_the_execution_columns(capsys):
    records = [
        record("tier:mid", "A", "pass", mode="route"),
        execute_record("A", "pass", "claim_matched"),
        execute_record("B", "fail", "corrected"),
        execute_record("C", "pass", "unverified"),
        execute_record("D", "pass", "executed"),
        execute_record("E", "pass", "executed"),
    ]

    run_eval.print_scorecard(run_eval.build_scorecard(records), case_count=5)
    printed = capsys.readouterr().out

    header = next(line for line in printed.splitlines() if line.startswith("strategy") and "p95 ms" in line)
    assert header.split()[-8:] == [
        "verified", "disputed", "one_sided", "claim_matched", "corrected", "executed", "unverified", "failed"
    ]
    baseline_row = next(line for line in printed.splitlines() if line.startswith("tier:mid"))
    execute_row = next(line for line in printed.splitlines() if line.startswith("execute:mid"))
    assert baseline_row.split()[-8:] == ["-"] * 8
    assert execute_row.split()[-8:] == ["0", "0", "0", "1", "1", "2", "1", "0"]
    assert "Execution (execute and crosscheck strategies" in printed


def test_scorecard_without_an_execute_strategy_has_no_execution_table(capsys):
    run_eval.print_scorecard(run_eval.build_scorecard([record("tier:mid", "A", "pass")]), case_count=1)

    assert "Execution (execute strategies" not in capsys.readouterr().out


def test_only_tier_strategies_ask_the_harness_not_to_fall_back():
    def no_fallback(strategy_text):
        return run_eval.build_request(run_eval.parse_strategy(strategy_text), "q", None).no_fallback

    assert no_fallback("tier:frontier") is True
    assert no_fallback("router") is False
    assert no_fallback("chain") is False
    assert no_fallback("execute:mid") is False


# ---- crosscheck:<tier A>+<tier B> strategy ----


def cross_result(status: str, answer: str | None, a: str | None, b: str | None, **extra):
    return fake_result(
        answer, mode="execute", execution_status=status, claimed_answer=None, executed_answer=a,
        executed_answer_b=b, cross_check=True, cross_check_tier="code",
        cross_check_rule="integer" if a is not None and b is not None else None, repaired=False, **extra,
    )


def run_cross_case(result, case_id: str = "CD-01", judge=None) -> dict:
    class CrossHarness:
        def answer(self, request):
            return result

    case = next(case for case in CASES if case["id"] == case_id)
    return run_eval.run_case(CrossHarness(), run_eval.parse_strategy("crosscheck:mid+code"), case, judge)


def test_parse_crosscheck_strategy():
    assert run_eval.parse_strategy("crosscheck:mid+code") == run_eval.Strategy(
        "crosscheck:mid+code", "crosscheck", "mid", "code"
    )
    assert run_eval.parse_strategy("crosscheck:mid") == run_eval.Strategy("crosscheck:mid", "crosscheck", "mid", "mid")
    assert "crosscheck" in run_eval.SUITE_KINDS[run_eval.SUITE_CLOSEDBOOK]
    assert "crosscheck" not in run_eval.SUITE_KINDS[run_eval.SUITE_OPENBOOK]
    for bad_text in ("crosscheck", "crosscheck:", "crosscheck:+code", "crosscheck:mid+"):
        with pytest.raises(ValueError):
            run_eval.parse_strategy(bad_text)


def test_crosscheck_strategy_requests_two_programs_on_its_two_tiers():
    request = run_eval.build_request(run_eval.parse_strategy("crosscheck:mid+code"), "q", None)

    assert (request.mode, request.tier, request.cross_check, request.cross_check_tier) == (
        "execute", "mid", True, "code"
    )
    assert run_eval.unsupported_reason(run_eval.parse_strategy("crosscheck:mid+code")) is None


def test_execute_strategy_measures_one_program_whatever_the_config_says():
    request = run_eval.build_request(run_eval.parse_strategy("execute:mid"), "q", None)

    assert request.cross_check is False and request.cross_check_tier is None


def test_a_verified_cross_check_is_graded_and_each_program_is_graded_too():
    row = run_cross_case(cross_result("verified", "Verified by two programs: ...\n\nAnswer: 2", "2", "2"))

    assert (row["outcome"], row["execution_status"]) == ("pass", "verified")
    assert (row["cross_check"], row["cross_check_tier"], row["cross_check_rule"]) == (True, "code", "integer")
    assert (row["executed_answer"], row["executed_answer_b"]) == ("2", "2")
    assert (row["program_a_grade"], row["program_b_grade"]) == ("pass", "pass")


def test_a_verified_cross_check_that_is_wrong_is_a_fail():
    row = run_cross_case(cross_result("verified", "Verified by two programs: ...\n\nAnswer: 3", "3", "3"))

    assert (row["outcome"], row["program_a_grade"], row["program_b_grade"]) == ("fail", "fail", "fail")


def test_a_disputed_cross_check_is_scored_as_a_fail_without_asking_the_grader():
    class ExplodingJudge:
        def judge(self, case, answer):
            raise AssertionError("a disputed run has no answer to adjudicate")

    disputed_text = "DISPUTED: ...\n\nProgram A printed: 2\nProgram B printed: 3\n\nNo answer is confirmed: the two programs disagree."
    row = run_cross_case(cross_result("disputed", disputed_text, "2", "3"), judge=ExplodingJudge())

    assert (row["outcome"], row["grade"]) == ("fail", "fail")
    assert row["grade_reason"].startswith("disputed:")
    assert (row["program_a_grade"], row["program_b_grade"]) == ("pass", "fail"), "A held the expected value"


def test_a_one_sided_cross_check_is_graded_on_the_program_that_ran():
    row = run_cross_case(cross_result("one_sided", "NOT CROSS-CHECKED: ...\n\nAnswer: 2", None, "2"))

    assert (row["outcome"], row["execution_status"]) == ("pass", "one_sided")
    assert (row["program_a_grade"], row["program_b_grade"]) == (None, "pass")


def test_programs_are_not_graded_one_by_one_on_a_rubric_case():
    disputed = cross_result("disputed", "DISPUTED: ...", "one thing", "another")
    row = run_cross_case(disputed, case_id="DM-01")

    assert row["outcome"] == "fail"
    assert (row["program_a_grade"], row["program_b_grade"]) == (None, None)


def test_a_disputed_record_stays_a_fail_when_regraded():
    disputed = dict(
        record("crosscheck:mid+code", "CD-01", "fail", mode="execute", execution_status="disputed"),
        answer="Program A printed: 2\nProgram B printed: 3", grade="fail", grade_reason="disputed: x",
    )

    regraded = run_eval.regrade_records([disputed], CASES, None)

    assert (regraded[0]["outcome"], regraded[0]["grade_reason"]) == ("fail", "disputed: x")


def cross_record(case_id: str, outcome: str, status: str, a_grade=None, b_grade=None, **extra) -> dict:
    return record(
        "crosscheck:mid+code", case_id, outcome, mode="execute", execution_status=status, cross_check=True,
        program_a_grade=a_grade, program_b_grade=b_grade, **extra,
    )


def test_summarise_execution_splits_a_cross_check_by_status_and_says_who_was_right():
    records = [
        cross_record("A", "pass", "verified", "pass", "pass"),
        cross_record("B", "fail", "verified", "fail", "fail"),
        cross_record("C", "fail", "disputed", "pass", "fail"),
        cross_record("D", "fail", "disputed", "pass", "fail"),
        cross_record("E", "fail", "disputed", "fail", "pass"),
        cross_record("F", "fail", "disputed", "fail", "fail"),
        cross_record("G", "fail", "disputed"),
        cross_record("H", "pass", "one_sided", "pass", None),
        cross_record("I", "error", "failed"),
    ]

    execution = run_eval.summarise_strategy("crosscheck:mid+code", records)["execution"]

    assert execution["verified"] == {"count": 2, "pass": 1, "graded": 2}
    assert execution["disputed"] == {"count": 5, "pass": 0, "graded": 5}
    assert execution["one_sided"] == {"count": 1, "pass": 1, "graded": 1}
    assert execution["failed"] == {"count": 1, "pass": 0, "graded": 0}
    assert execution["disputed_programs"] == {
        "a_right": 2, "b_right": 1, "both_right": 0, "neither_right": 1, "not_graded_per_program": 1
    }


def test_scorecard_prints_the_disputed_table_for_a_cross_check_only(capsys):
    records = [cross_record("A", "pass", "verified", "pass", "pass"), cross_record("C", "fail", "disputed", "pass", "fail")]

    run_eval.print_scorecard(run_eval.build_scorecard(records), case_count=2)
    printed = capsys.readouterr().out

    assert "Disputed cross-checks: which program held the expected value" in printed
    after_heading = printed.split("Disputed cross-checks", 1)[1]
    row = next(line for line in after_heading.splitlines() if line.startswith("crosscheck:mid+code"))
    assert row.split()[1:] == ["1", "1", "0", "0", "0", "0"]

    run_eval.print_scorecard(
        run_eval.build_scorecard([execute_record("A", "pass", "executed")]), case_count=1
    )
    assert "Disputed cross-checks" not in capsys.readouterr().out


# ---- auto and auto:check strategies ----


def test_parse_auto_strategies():
    assert run_eval.parse_strategy("auto") == run_eval.Strategy("auto", "auto")
    assert run_eval.parse_strategy("auto:check") == run_eval.Strategy("auto:check", "auto:check")
    for kind in ("auto", "auto:check"):
        assert kind in run_eval.SUITE_KINDS[run_eval.SUITE_CLOSEDBOOK]
        assert kind not in run_eval.SUITE_KINDS[run_eval.SUITE_OPENBOOK]
        assert run_eval.unsupported_reason(run_eval.parse_strategy(kind)) is None
    for bad_text in ("auto:", "auto:mid", "autocheck"):
        with pytest.raises(ValueError):
            run_eval.parse_strategy(bad_text)


def test_auto_strategies_request_auto_mode_with_the_check_forced_off_or_on():
    plain = run_eval.build_request(run_eval.parse_strategy("auto"), "q", None)
    checked = run_eval.build_request(run_eval.parse_strategy("auto:check"), "q", None)

    assert (plain.mode, plain.tier, plain.computation_check) == ("auto", None, False)
    assert (checked.mode, checked.tier, checked.computation_check) == ("auto", None, True)
    assert plain.no_fallback is False


def test_computation_check_fields_are_recorded_per_case():
    class CheckingHarness:
        def answer(self, request):
            return fake_result(
                "Verified by two programs: ...\n\nAnswer: 2", mode="execute", execution_status="verified",
                auto_reason="auto: ... [check on triage: yes, confidence 0.90]", computation_verdict="yes",
                computation_check_tier="triage", computation_check_confidence=0.9,
                calls=[
                    SimpleNamespace(role="check", tier="triage", latency_ms=240.0, cost_usd=0.0, error_message=None),
                    SimpleNamespace(role="draft", tier="mid", latency_ms=9000.0, cost_usd=0.0, error_message=None),
                ],
            )

    case = {**next(case for case in CASES if case["id"] == "CD-01"), "computable": True}
    row = run_eval.run_case(CheckingHarness(), run_eval.parse_strategy("auto:check"), case, None)

    assert row["outcome"] == "pass"
    assert (row["computation_verdict"], row["computation_check_tier"], row["computation_check_confidence"]) == (
        "yes", "triage", 0.9
    )
    assert row["computation_check_latency_ms"] == 240.0
    assert row["expected_computable"] is True
    assert row["auto_reason"].endswith("confidence 0.90]")


def test_a_case_without_a_label_records_none():
    assert run_eval.blank_record(run_eval.parse_strategy("auto"), CASES[0])["expected_computable"] is None


def check_record(case_id, verdict, expected, category="reasoning", mode="route", latency_ms=200.0, outcome="pass"):
    return record(
        "auto:check", case_id, outcome, category=category, mode=mode, computation_verdict=verdict,
        expected_computable=expected, computation_check_latency_ms=latency_ms,
        execution_status="verified" if mode == "execute" else "skipped",
    )


def test_summarise_computation_check():
    records = [
        check_record("A", "yes", True, mode="execute", latency_ms=100.0),
        check_record("B", "yes", True, mode="route", latency_ms=110.0),
        check_record("C", "yes", False, category="lookup", mode="execute", latency_ms=120.0),
        check_record("D", "no", True, latency_ms=300.0),
        check_record("E", "no", False, category="lookup", latency_ms=200.0),
        check_record("F", "low_confidence", True, latency_ms=250.0),
        check_record("G", "failed", False, category="domain", latency_ms=5000.0),
        check_record("H", "no", None, category="domain", latency_ms=None),
        record("auto", "A", "pass", mode="route", computation_verdict="not_run", expected_computable=True),
    ]

    check = run_eval.summarise_strategy("auto:check", records)["computation_check"]

    assert check["checked"] == 8
    assert check["verdicts"] == {"yes": 3, "no": 3, "low_confidence": 1, "failed": 1}
    assert check["sent_to_execute"] == 3
    assert check["sent_to_execute_by_category"] == {"reasoning": 2, "lookup": 1}
    assert check["execute_fallbacks"] == ["B"]
    assert check["labelled"] == 7
    assert (check["true_positive"], check["true_negative"]) == (2, 2)
    assert check["false_positive_cases"] == ["C"]
    assert check["false_negative_cases"] == ["D", "F"]
    assert check["added_latency_p50_ms"] == 250.0
    assert check["added_latency_p95_ms"] == 5000.0
    assert run_eval.summarise_strategy("auto", records)["computation_check"] is None


def test_scorecard_prints_the_computation_check_table_only_when_the_check_ran(capsys):
    records = [
        record("auto", "A", "pass", mode="route", computation_verdict="not_run"),
        check_record("A", "yes", True, mode="execute"),
        check_record("B", "no", False, category="lookup"),
    ]

    run_eval.print_scorecard(run_eval.build_scorecard(records), case_count=2)
    printed = capsys.readouterr().out

    assert "Computation check (auto:check" in printed
    check_row = [line for line in printed.splitlines() if line.startswith("auto:check")][-2]
    assert "reasoning 1" in check_row

    run_eval.print_scorecard(run_eval.build_scorecard(records[:1]), case_count=1)
    assert "Computation check (auto:check" not in capsys.readouterr().out


# --- the code set: reference tests in the sandbox, and what the generated tests were worth ---

from inference_arbiter import codetest, codetest_driver  # noqa: E402
from inference_arbiter.schemas import AskRequest  # noqa: E402

from conftest import (  # noqa: E402
    HarnessKit,
    make_execute_config,
    ok_result,
    program_output,
    run_program_on_host,
    sandbox_result,
)

CODE_CASES, _ = run_eval.load_golden(REPO_ROOT / "eval" / "code_questions.yaml")
LUHN_CASE = next(case for case in CODE_CASES if case["id"] == "CQ-01")
LUHN_GOOD = (REPO_ROOT / LUHN_CASE["reference_solution"]).read_text(encoding="utf-8")
LUHN_WRONG = (REPO_ROOT / LUHN_CASE["broken_solutions"][0]).read_text(encoding="utf-8")
# Generated tests that hold one right expected value: they pass a right solution and
# happen to pass the first seeded bug too.
LUHN_WEAK_TESTS = '''import unittest
from solution import luhn_check_digit


class T(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(luhn_check_digit("0"), 0)
'''
LUHN_STRONG_TESTS = '''import unittest
from solution import luhn_check_digit


class T(unittest.TestCase):
    def test_known(self):
        self.assertEqual(luhn_check_digit("7992739871"), 3)

    def test_zero(self):
        self.assertEqual(luhn_check_digit("0"), 0)
'''
# Generated tests with a wrong expected value: they fail a right solution.
LUHN_WRONG_TESTS = '''import unittest
from solution import luhn_check_digit


class T(unittest.TestCase):
    def test_known(self):
        self.assertEqual(luhn_check_digit("7992739871"), 4)
'''


def host_runner(tmp_path):
    programs: list[str] = []

    def runner(program: str):
        programs.append(program)
        return run_program_on_host(tmp_path, program)

    runner.programs = programs
    return runner


def driver_output(stage: str = "ran", tests_run: int = 2, **fields):
    """What the harness's own sandbox run returns for a driver run with these counts."""
    summary = codetest_driver.blank_summary(stage)
    failed = any(fields.get(name) for name in ("failures", "errors"))
    summary.update(tests_run=tests_run, stub_passed=None if failed or stage != "ran" else False)
    summary.update(fields)
    return program_output(
        stdout=codetest.RESULT_MARKER + json.dumps(summary) + "\n", exit_code=codetest_driver.exit_code_for(summary)
    )


def code_kit(**code_test_overrides) -> HarnessKit:
    settings = {"interface_tier": "local", "solution_tier": "mid", "tests_tier": "code", "parallel": False}
    settings.update(code_test_overrides)
    return HarnessKit(make_execute_config(code_test=settings))


def script_codetest(kit: HarnessKit, tests: str, *solutions: str, runs) -> None:
    signature = "def luhn_check_digit(payload: str) -> int:"
    kit.client.script("local-model", ok_result(json.dumps({"function_name": "luhn_check_digit", "signature": signature})))
    kit.client.script("mid-model", *[ok_result(json.dumps({"working": "w", "solution": text})) for text in solutions])
    kit.client.script("code-model", ok_result(json.dumps({"working": "w", "tests": tests})))
    kit.sandbox_runner.script("run", *runs)


CODETEST_STRATEGY = run_eval.parse_strategy("codetest:mid+code")


def test_parse_codetest_strategy():
    both = run_eval.parse_strategy("codetest:mid+local")
    assert (both.kind, both.tier, both.tests_tier) == ("codetest", "mid", "local")
    one = run_eval.parse_strategy("codetest:mid")
    assert (one.kind, one.tier, one.tests_tier) == ("codetest", "mid", None)
    for bad in ("codetest:", "codetest:+local", "codetest:mid+"):
        with pytest.raises(ValueError, match="codetest:<solution tier>"):
            run_eval.parse_strategy(bad)
    assert "codetest" in run_eval.SUITE_KINDS[run_eval.SUITE_CLOSEDBOOK]
    assert run_eval.unsupported_reason(both) is None


def test_codetest_strategy_requests_codetest_mode_on_its_tiers():
    request = run_eval.build_request(run_eval.parse_strategy("codetest:mid+local"), "q", None)
    assert (request.mode, request.tier, request.tests_tier) == ("codetest", "mid", "local")
    default_tests = run_eval.build_request(run_eval.parse_strategy("codetest:mid"), "q", None)
    assert (default_tests.mode, default_tests.tests_tier) == ("codetest", None)


@pytest.mark.parametrize(
    "tests_status, reference_grade, verdict",
    [
        ("tests_failed", "fail", "caught"),
        ("tests_passed", "fail", "missed"),
        ("tests_failed", "pass", "false_alarm"),
        ("tests_passed", "pass", "clean"),
        ("untested", "pass", "untested"),
        ("untested", "fail", "untested"),
        ("tests_passed", "ungraded", None),
        ("tests_failed", "ungraded", None),
        ("untested", "ungraded", None),
        ("tests_passed", None, None),
        ("failed", "fail", None),
        ("skipped", "pass", None),
        (None, "pass", None),
    ],
)
def test_model_tests_verdict_table(tests_status, reference_grade, verdict):
    assert run_eval.model_tests_verdict(tests_status, reference_grade) == verdict


@pytest.mark.parametrize(
    "run_class, grade",
    [
        ("passed", "pass"), ("tests_failed", "fail"), ("solution_invalid", "fail"), ("not_finished", "fail"),
        ("tests_invalid", "fail"), ("vacuous", "fail"), ("infra_failed", "ungraded"),
    ],
)
def test_reference_grade_of(run_class, grade):
    run = {"run_class": run_class, "reason": "why", "tests_run": 6, "tests_failed": 2}
    assert run_eval.reference_grade_of(run)[0] == grade


def test_answer_code_is_the_last_code_fence():
    answer = "First try:\n```python\nx = 1\n```\nBetter:\n```python\ndef f():\n    return 2\n```\nDone."
    assert run_eval.answer_code(answer) == "def f():\n    return 2\n"
    assert run_eval.answer_code("no code here") is None
    assert run_eval.answer_code(None) is None


@pytest.mark.parametrize(
    "answer, outcome, reason_part",
    [
        (f"Here it is.\n\n```python\n{LUHN_GOOD}```", "pass", "reference tests passed"),
        (f"```python\n{LUHN_WRONG}```", "fail", "reference tests failed"),
        ("```python\ndef luhn_check_digit(payload)\n```", "fail", "solution_invalid"),
        ("```python\ndef other(payload):\n    return 0\n```", "fail", "solution_missing_function"),
        ("The check digit is computed by doubling every second digit.", "fail", "no code block"),
    ],
)
def test_a_plain_tier_answer_on_a_code_case_is_graded_by_the_reference_tests(tmp_path, answer, outcome, reason_part):
    requests: list = []
    harness = SimpleNamespace(
        answer=lambda request: (requests.append(request), fake_result(answer, mode="route"))[1]
    )
    judge = FakeJudge()

    record = run_eval.run_case(
        harness, run_eval.parse_strategy("tier:mid"), LUHN_CASE, judge, reference_runner=host_runner(tmp_path)
    )

    assert record["outcome"] == outcome and reason_part in record["grade_reason"]
    assert record["match_type"] == "reference_tests"
    assert judge.judged == [], "no judge is involved in a code case"
    assert record["model_tests_verdict"] is None and record["seeded"] is None
    assert requests[0].messages[0].content == LUHN_CASE["question"]


def test_without_a_sandbox_runner_a_code_case_is_ungraded():
    harness = SimpleNamespace(answer=lambda request: fake_result(f"```python\n{LUHN_GOOD}```", mode="route"))

    record = run_eval.run_case(harness, run_eval.parse_strategy("tier:mid"), LUHN_CASE, None)

    assert record["outcome"] == "ungraded" and "no sandbox runner" in record["grade_reason"]


def test_a_sandbox_failure_during_reference_grading_is_ungraded():
    harness = SimpleNamespace(answer=lambda request: fake_result(f"```python\n{LUHN_GOOD}```", mode="route"))

    record = run_eval.run_case(
        harness, run_eval.parse_strategy("tier:mid"), LUHN_CASE, None,
        reference_runner=lambda program: sandbox_result("sandbox_unavailable", exit_code=None),
    )

    assert record["outcome"] == "ungraded"
    assert record["reference_run"]["run_class"] == "infra_failed"


@pytest.mark.parametrize(
    "solution, tests, own_run, status, outcome, verdict",
    [
        (LUHN_WRONG, LUHN_WEAK_TESTS, driver_output(tests_run=1), "tests_passed", "fail", "missed"),
        (LUHN_GOOD, LUHN_STRONG_TESTS, driver_output(), "tests_passed", "pass", "clean"),
        (LUHN_GOOD, LUHN_STRONG_TESTS, driver_output(stub_passed=True), "untested", "pass", "untested"),
    ],
)
def test_codetest_runs_are_graded_by_the_reference_tests_not_by_their_own(
    tmp_path, solution, tests, own_run, status, outcome, verdict
):
    kit = code_kit()
    script_codetest(kit, tests, solution, runs=[own_run])

    record = run_eval.run_case(kit.harness, CODETEST_STRATEGY, LUHN_CASE, None, reference_runner=host_runner(tmp_path))

    assert (record["execution_status"], record["outcome"], record["model_tests_verdict"]) == (status, outcome, verdict)
    assert record["first_model_tests_verdict"] == verdict, "no repair: the first solution is the returned one"
    assert record["first_reference_run"] is None
    assert (record["mode"], record["tests_tier"], record["code_function"]) == ("codetest", "code", "luhn_check_digit")
    assert (record["code_solution"], record["code_tests"]) == (solution, tests)


def test_caught_and_false_alarm_come_from_a_failing_generated_suite(tmp_path):
    failed_run = driver_output(failures=1, failed_tests=["test_known"])
    caught_kit, alarm_kit = code_kit(max_repairs=0), code_kit(max_repairs=0)
    script_codetest(caught_kit, LUHN_STRONG_TESTS, LUHN_WRONG, runs=[failed_run])
    script_codetest(alarm_kit, LUHN_WRONG_TESTS, LUHN_GOOD, runs=[failed_run])
    runner = host_runner(tmp_path)

    caught = run_eval.run_case(caught_kit.harness, CODETEST_STRATEGY, LUHN_CASE, None, reference_runner=runner)
    alarm = run_eval.run_case(alarm_kit.harness, CODETEST_STRATEGY, LUHN_CASE, None, reference_runner=runner)

    assert (caught["execution_status"], caught["outcome"], caught["model_tests_verdict"]) == ("tests_failed", "fail", "caught")
    assert (alarm["execution_status"], alarm["outcome"], alarm["model_tests_verdict"]) == ("tests_failed", "pass", "false_alarm")
    assert alarm["seeded"]["reference_solution"]["run_class"] == "tests_failed", "wrong tests reject the reference solution"


def test_first_and_repaired_solutions_are_recorded_separately(tmp_path):
    kit = code_kit()
    script_codetest(
        kit, LUHN_STRONG_TESTS, LUHN_WRONG, LUHN_GOOD,
        runs=[driver_output(failures=1, failed_tests=["test_known"]), driver_output()],
    )

    record = run_eval.run_case(kit.harness, CODETEST_STRATEGY, LUHN_CASE, None, reference_runner=host_runner(tmp_path))

    assert (record["repaired"], record["code_first_outcome"]) == (True, "tests_failed")
    assert (record["outcome"], record["model_tests_verdict"]) == ("pass", "clean")
    assert (record["first_reference_grade"], record["first_model_tests_verdict"]) == ("fail", "caught")
    assert record["first_reference_run"]["run_class"] == "tests_failed"
    assert (record["code_first_solution"], record["code_solution"]) == (LUHN_WRONG, LUHN_GOOD)
    summary = run_eval.summarise_code_tests([record])
    assert (summary["repaired"], summary["repair_fixed"], summary["repair_broke"]) == (1, 1, 0)
    assert summary["first_attempt"]["caught"] == 1 and summary["returned"]["clean"] == 1


def test_seeded_bugs_are_run_against_the_generated_tests(tmp_path):
    weak_kit, strong_kit = code_kit(), code_kit()
    script_codetest(weak_kit, LUHN_WEAK_TESTS, LUHN_GOOD, runs=[driver_output(tests_run=1)])
    script_codetest(strong_kit, LUHN_STRONG_TESTS, LUHN_GOOD, runs=[driver_output()])
    runner = host_runner(tmp_path)

    weak = run_eval.run_case(weak_kit.harness, CODETEST_STRATEGY, LUHN_CASE, None, reference_runner=runner)
    strong = run_eval.run_case(strong_kit.harness, CODETEST_STRATEGY, LUHN_CASE, None, reference_runner=runner)

    assert [run["path"] for run in weak["seeded"]["broken"]] == LUHN_CASE["broken_solutions"]
    assert [run["run_class"] for run in weak["seeded"]["broken"]] == ["passed", "tests_failed"]
    assert [run["run_class"] for run in strong["seeded"]["broken"]] == ["tests_failed", "tests_failed"]
    assert weak["seeded"]["reference_solution"]["run_class"] == "passed"
    seeded = run_eval.summarise_code_tests([weak, strong])["seeded"]
    assert (seeded["cases_with_tests"], seeded["broken_runs"], seeded["caught"], seeded["missed"]) == (2, 4, 3, 1)
    assert seeded["catch_rate"]["rate"] == 0.75
    assert (seeded["tests_pass_reference_solution"], seeded["tests_reject_reference_solution"]) == (2, 0)


def test_a_codetest_run_without_an_answer_is_an_error_and_is_not_graded(tmp_path):
    kit = code_kit()
    kit.client.script("local-model", ok_result("not json"))
    runner = host_runner(tmp_path)

    record = run_eval.run_case(kit.harness, CODETEST_STRATEGY, LUHN_CASE, None, reference_runner=runner)

    assert (record["outcome"], record["error_type"]) == ("error", "interface_invalid")
    assert runner.programs == [] and record["reference_run"] is None


def code_record(verdict, first_verdict=None, outcome="pass", **extra) -> dict:
    fields = dict(
        mode="codetest", model_tests_verdict=verdict,
        first_model_tests_verdict=first_verdict if first_verdict is not None else verdict,
    )
    fields.update(extra)
    return record("codetest:mid+local", f"CQ-{len(fields)}", outcome, **fields)


def test_code_tests_summary_counts_and_rates():
    records = (
        [code_record("caught", outcome="fail")] * 3
        + [code_record("missed", outcome="fail")]
        + [code_record("false_alarm")]
        + [code_record("clean")] * 4
        + [code_record("untested")]
        + [code_record(None, outcome="error")]
    )

    summary = run_eval.summarise_code_tests(records)

    returned = summary["returned"]
    assert (returned["caught"], returned["missed"], returned["false_alarm"], returned["clean"]) == (3, 1, 1, 4)
    assert (returned["untested"], returned["no_verdict"], summary["runs"]) == (1, 1, 11)
    assert (returned["catch_rate"]["count"], returned["catch_rate"]["of"], returned["catch_rate"]["rate"]) == (3, 4, 0.75)
    assert (returned["false_alarm_rate"]["count"], returned["false_alarm_rate"]["of"]) == (1, 5)
    assert returned["catch_rate"]["ci95"] == list(run_eval.wilson_interval(3, 4))
    assert summary["seeded"]["catch_rate"]["rate"] is None, "no seeded run, no rate"


def test_strategies_that_are_not_codetest_have_no_code_tests_summary(capsys):
    records = [record("tier:mid", "CQ-01", "pass"), code_record("clean")]

    scorecard = run_eval.build_scorecard(records)
    run_eval.print_scorecard(scorecard, 1)

    by_name = {summary["strategy"]: summary for summary in scorecard["strategies"]}
    assert by_name["tier:mid"]["code_tests"] is None
    assert by_name["codetest:mid+local"]["code_tests"]["runs"] == 1
    printed = capsys.readouterr().out
    assert "Generated tests against the reference tests" in printed
    assert "Repairs and seeded bugs" in printed
    assert printed.count("codetest:mid+local") >= 3


def test_regrade_keeps_a_reference_graded_record_as_it_is():
    stored = record("tier:mid", "CQ-01", "fail", answer="```python\nx\n```", match_type="reference_tests",
                    grade="fail", grade_reason="2 of 6 reference tests failed")

    (regraded,) = run_eval.regrade_records([stored], [LUHN_CASE], None)

    assert (regraded["outcome"], regraded["grade_reason"]) == ("fail", "2 of 6 reference tests failed")


# --- guardrail: nothing of the reference files reaches a model call ---

REFERENCE_MARK = "REFERENCE-ONLY-MARKER-91c2"


def marked_case(tmp_path) -> dict:
    """A code case whose three reference files all carry the marker."""
    files = {
        "tests.py": f"# {REFERENCE_MARK} tests\n{LUHN_STRONG_TESTS}",
        "good.py": f"# {REFERENCE_MARK} solution\n{LUHN_GOOD}",
        "broken.py": f"# {REFERENCE_MARK} broken\n{LUHN_WRONG}",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    case = dict(LUHN_CASE)
    case["match"] = {**LUHN_CASE["match"], "tests": str(tmp_path / "tests.py")}
    case["reference_solution"] = str(tmp_path / "good.py")
    case["broken_solutions"] = [str(tmp_path / "broken.py")]
    return case


def assert_no_reference_text_in_any_call(kit: HarnessKit) -> None:
    assert kit.client.calls, "the scenario made model calls"
    for call in kit.client.calls:
        for message in call.messages:
            assert REFERENCE_MARK not in message["content"], "reference text reached a model call"


def run_marked_codetest(tmp_path, kit: HarnessKit, runner):
    script_codetest(
        kit, LUHN_WEAK_TESTS, LUHN_WRONG, LUHN_GOOD,
        runs=[driver_output(failures=1, failed_tests=["test_zero"]), driver_output(tests_run=1)],
    )
    return run_eval.run_case(kit.harness, CODETEST_STRATEGY, marked_case(tmp_path), None, reference_runner=runner)


def test_no_reference_text_reaches_any_call_of_a_codetest_run(tmp_path):
    kit, runner = code_kit(), host_runner(tmp_path)

    record = run_marked_codetest(tmp_path, kit, runner)

    assert_no_reference_text_in_any_call(kit)
    assert [call.model for call in kit.client.calls] == ["local-model", "mid-model", "code-model", "mid-model"]
    assert record["reference_run"] is not None and record["first_reference_run"] is not None
    marked_programs = [program for program in runner.programs if REFERENCE_MARK in program]
    assert len(marked_programs) == 4, "the marked files did go to the sandbox: two reference runs and two seeded runs"
    assert all(REFERENCE_MARK not in str(value) for value in record.values()), "nor into the stored record"


def test_no_reference_text_reaches_the_call_of_a_plain_tier_run(tmp_path):
    kit, runner = HarnessKit(make_execute_config()), host_runner(tmp_path)
    kit.client.script("mid-model", ok_result(f"```python\n{LUHN_GOOD}```"))

    record = run_eval.run_case(
        kit.harness, run_eval.parse_strategy("tier:mid"), marked_case(tmp_path), None, reference_runner=runner
    )

    assert record["outcome"] == "pass"
    assert_no_reference_text_in_any_call(kit)
    assert kit.client.calls[0].messages == [{"role": "user", "content": LUHN_CASE["question"]}]
    assert len(runner.programs) == 1 and REFERENCE_MARK in runner.programs[0]


def test_the_reference_check_catches_a_leak(tmp_path, monkeypatch):
    """The check is not vacuous: a request built with the reference tests in it fails it."""
    honest_builder = run_eval.build_request

    def leaky_builder(strategy, question, verifier_tier):
        request = honest_builder(strategy, question, verifier_tier)
        leaked = (tmp_path / "tests.py").read_text(encoding="utf-8")
        return AskRequest.from_question(
            f"{question}\n\nThe tests are:\n{leaked}", mode=request.mode, tier=request.tier,
            tests_tier=request.tests_tier,
        )

    monkeypatch.setattr(run_eval, "build_request", leaky_builder)
    kit = code_kit()
    run_marked_codetest(tmp_path, kit, host_runner(tmp_path))

    with pytest.raises(AssertionError, match="reference text reached a model call"):
        assert_no_reference_text_in_any_call(kit)


def test_reference_files_are_read_only_after_the_answer_came_back(tmp_path, monkeypatch):
    events: list[str] = []
    honest_read = run_eval.read_reference_file
    monkeypatch.setattr(
        run_eval, "read_reference_file", lambda path: (events.append("read"), honest_read(path))[1]
    )
    kit = code_kit()
    script_codetest(kit, LUHN_STRONG_TESTS, LUHN_GOOD, runs=[driver_output()])
    honest_answer = kit.harness.answer
    kit.harness.answer = lambda request: (honest_answer(request), events.append("answered"))[0]

    run_eval.run_case(kit.harness, CODETEST_STRATEGY, marked_case(tmp_path), None, reference_runner=host_runner(tmp_path))

    assert events[0] == "answered" and events.count("answered") == 1
    assert events.count("read") == 3, "the reference tests, the reference solution and the one seeded bug"
