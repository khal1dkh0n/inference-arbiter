"""Auto mode: the decision table, the harness wiring and the CLI flags. Fakes only."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from inference_arbiter import auto, cli, openbook
from inference_arbiter.config import HarnessConfig
from inference_arbiter.router import OpenbookNotConfiguredError
from inference_arbiter.schemas import AskRequest

from conftest import BASE_CONFIG, HarnessKit, error_result, make_config, make_openbook_config, ok_result, triage_json

QUESTION = "What confidence value triggers escalation?"
GUIDE_TEXT = "# Router Guide\n\nThe router escalates when confidence is below 0.80.\n"
QUOTE = "The router escalates when confidence is below 0.80."


# --- the decision table ---


@pytest.mark.parametrize(
    ("has_sources", "estimated", "window", "ratio", "require_citations", "mode", "reason"),
    [
        # row 1: no sources
        (False, 0, None, 0.8, False, "route", "auto: no sources, one closed-book call"),
        (False, 0, 1_000_000, 0.8, False, "route", "auto: no sources, one closed-book call"),
        # row 2: citations required wins over a fit
        (True, 100, 1_000_000, 0.8, True, "openbook", "auto: citations required"),
        (True, 100, None, 0.8, True, "openbook", "auto: citations required"),
        # row 3: unknown window is treated as not fitting
        (
            True,
            100,
            None,
            0.8,
            False,
            "openbook",
            "auto: the answering tier has no context_window_tokens, treated as not fitting",
        ),
        # row 4: fits, the limit itself included
        (True, 17_000, 1_048_576, 0.8, False, "stuffed", "auto: sources fit, 17k of 839k tokens"),
        (True, 800, 1000, 0.8, False, "stuffed", "auto: sources fit, 800 of 800 tokens"),
        # row 5: does not fit, the safety margin included
        (True, 801, 1000, 0.8, False, "openbook", "auto: sources exceed context window, 801 of 800 tokens"),
        (
            True,
            3_400_000,
            1_048_576,
            0.8,
            False,
            "openbook",
            "auto: sources exceed context window, 3.4M of 839k tokens",
        ),
        (True, 900, 1000, 1.0, False, "stuffed", "auto: sources fit, 900 of 1k tokens"),
    ],
)
def test_decision_table(has_sources, estimated, window, ratio, require_citations, mode, reason):
    decision = auto.decide_mode(has_sources, estimated, window, ratio, require_citations)

    assert (decision.mode, decision.reason) == (mode, reason)


# --- request contract ---


def test_auto_is_the_default_mode_and_accepts_sources():
    assert AskRequest.from_question(QUESTION).mode == "auto"
    assert AskRequest.from_question(QUESTION, sources=["a.md"]).mode == "auto"


def test_require_citations_without_sources_is_refused():
    with pytest.raises(ValidationError, match="nothing to cite"):
        AskRequest.from_question(QUESTION, require_citations=True)


@pytest.mark.parametrize("mode", ["stuffed", "openbook"])
def test_require_citations_only_fits_auto_and_openbook(mode):
    if mode == "openbook":
        assert AskRequest.from_question(QUESTION, mode=mode, sources=["a.md"], require_citations=True)
        return
    with pytest.raises(ValidationError, match="require_citations needs mode"):
        AskRequest.from_question(QUESTION, mode=mode, sources=["a.md"], require_citations=True)


def test_triage_timeout_simulation_is_refused_in_auto_mode():
    with pytest.raises(ValidationError, match="needs a mode with a triage call"):
        AskRequest.from_question(QUESTION, simulate="triage-timeout")


def test_verifier_tier_in_auto_needs_sources():
    assert AskRequest.from_question(QUESTION, sources=["a.md"], verifier_tier="ground_chat")
    with pytest.raises(ValidationError, match="verifier_tier needs"):
        AskRequest.from_question(QUESTION, verifier_tier="ground_chat")


# --- config ---


def test_closed_book_tier_defaults_to_the_fallback_tier():
    config = make_config()

    assert config.auto.closed_book_tier is None
    assert config.auto_closed_book_tier() == "mid"
    assert config.auto.stuffed_fill_ratio == 0.8


def test_closed_book_tier_must_be_an_answer_tier():
    with pytest.raises(ValidationError, match="auto.closed_book_tier 'triage' is not an answer tier"):
        make_config(auto={"closed_book_tier": "triage"})


def test_shipped_config_names_the_closed_book_tier():
    from inference_arbiter.config import load_config

    config = load_config("config/models.yaml")

    assert config.auto.closed_book_tier == "mid"
    assert config.tiers["mid"].context_window_tokens is not None


# --- harness: no sources ---


def test_no_sources_is_one_call_to_the_closed_book_tier_without_triage():
    kit = HarnessKit(make_config(auto={"closed_book_tier": "local"}))
    kit.client.script("local-model", ok_result("a nonce is used once"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?"))

    assert kit.client.models_called() == ["local-model"]
    assert result.mode == "route"
    assert result.auto_reason == "auto: no sources, one closed-book call"
    assert result.triage_status == "skipped"
    assert (result.selected_tier, result.route_reason) == ("local", "auto")
    assert [(call.role, call.tier) for call in result.calls] == [("answer", "local")]
    assert result.answer == "a nonce is used once"
    assert kit.run_record.mode == "route"
    assert kit.run_record.auto_reason == result.auto_reason


def test_no_sources_keeps_the_on_error_fallback_hop():
    kit = HarnessKit(make_config())
    kit.client.script("mid-model", error_result("http_5xx"))
    kit.client.script("local-model", ok_result("from the fallback"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?"))

    assert kit.client.models_called() == ["mid-model", "local-model"]
    assert result.fallback_used is True
    assert result.final_tier == "local"
    assert result.answer == "from the fallback"


def test_forced_tier_wins_over_the_closed_book_tier():
    kit = HarnessKit(make_config())
    kit.client.script("local-model", ok_result("forced"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?", tier="local"))

    assert kit.client.models_called() == ["local-model"]
    assert (result.selected_tier, result.route_reason) == ("local", "forced")
    assert result.auto_reason == "auto: no sources, one closed-book call"


def test_explicit_route_mode_still_triages_and_reports_no_auto_reason():
    kit = HarnessKit(make_config())
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("local-model", ok_result("routed"))

    result = kit.harness.answer(AskRequest.from_question("what is a nonce?", mode="route"))

    assert kit.client.models_called() == ["triage-model", "local-model"]
    assert result.route_reason == "triage_label"
    assert result.auto_reason is None
    assert kit.run_record.auto_reason is None


# --- harness: sources ---


def windowed_config(tmp_path, window: int | None, **openbook_overrides) -> HarnessConfig:
    config = make_openbook_config(tmp_path / "cache", openbook_overrides)
    config.tiers["mid"].context_window_tokens = window
    return config


@pytest.fixture
def guide_path(tmp_path) -> str:
    path = tmp_path / "guide.md"
    path.write_text(GUIDE_TEXT, encoding="utf-8")
    return str(path)


def cited_draft() -> object:
    claim = {"text": "Escalation is below 0.80.", "passage_id": "S1-P1", "quote": QUOTE}
    return ok_result(
        json.dumps({"working": "read", "answerable": True, "answer": "Below 0.80.", "claims": [claim]})
    )


def grounded_report() -> object:
    return ok_result(
        json.dumps(
            {"claims": [{"claim_number": 1, "reason": "stated", "verdict": "supported"}], "answerable": "yes"}
        )
    )


def test_sources_that_fit_go_stuffed_in_one_call(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", ok_result("Below 0.80."))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert kit.client.models_called() == ["mid-model"]
    assert result.mode == "stuffed"
    assert result.auto_reason.startswith("auto: sources fit, ")
    assert result.auto_reason.endswith(" of 8k tokens")
    assert (result.selected_tier, result.route_reason) == ("mid", "auto")
    assert result.triage_status == "skipped"
    assert result.citations == []
    assert GUIDE_TEXT.strip() in kit.client.calls[0].messages[1]["content"]
    assert kit.run_record.mode == "stuffed"
    assert kit.run_record.auto_reason == result.auto_reason


def test_sources_that_do_not_fit_go_through_the_pipeline(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 20))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert kit.client.models_called() == ["mid-model", "ground-model"]
    assert result.mode == "openbook"
    assert result.auto_reason.startswith("auto: sources exceed context window, ")
    assert result.route_reason == "configured"
    assert result.grounding_status == "grounded"
    assert [citation.passage_id for citation in result.citations] == ["S1-P1"]


def test_unknown_context_window_is_treated_as_not_fitting(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, None))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert result.mode == "openbook"
    assert result.auto_reason == auto.REASON_NO_WINDOW


def test_require_citations_takes_the_pipeline_even_when_the_sources_fit(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    result = kit.harness.answer(
        AskRequest.from_question(QUESTION, sources=[guide_path], require_citations=True)
    )

    assert result.mode == "openbook"
    assert result.auto_reason == "auto: citations required"
    assert len(result.citations) == 1


def test_the_window_checked_is_the_forced_tiers(tmp_path, guide_path):
    config = windowed_config(tmp_path, 20)
    config.tiers["local"].context_window_tokens = 10_000
    kit = HarnessKit(config)
    kit.client.script("local-model", ok_result("Below 0.80."))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path], tier="local"))

    assert kit.client.models_called() == ["local-model"]
    assert result.mode == "stuffed"
    assert result.route_reason == "forced"


def test_sources_are_loaded_once_and_reused(tmp_path, guide_path, monkeypatch):
    load_calls = []
    real_load_sources = openbook.load_sources

    def counting_load_sources(sources, **settings):
        load_calls.append(list(sources))
        return real_load_sources(sources, **settings)

    monkeypatch.setattr(openbook, "load_sources", counting_load_sources)
    stuffed_kit = HarnessKit(windowed_config(tmp_path, 10_000))
    stuffed_kit.client.script("mid-model", ok_result("Below 0.80."))
    pipeline_kit = HarnessKit(windowed_config(tmp_path, 20))
    pipeline_kit.client.script("mid-model", cited_draft())
    pipeline_kit.client.script("ground-model", grounded_report())

    stuffed_kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))
    pipeline_kit.harness.answer(AskRequest.from_question(QUESTION, sources=[guide_path]))

    assert load_calls == [[guide_path], [guide_path]]


def test_a_source_that_cannot_be_loaded_fails_the_run_before_any_model_call(tmp_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))

    result = kit.harness.answer(AskRequest.from_question(QUESTION, sources=[str(tmp_path / "absent.md")]))

    assert kit.client.models_called() == []
    assert result.status == "error"
    assert result.error_type == "source_load_failed"
    assert result.auto_reason == auto.REASON_LOAD_FAILED


def test_sources_without_an_openbook_block_are_a_usage_error():
    kit = HarnessKit(HarnessConfig.model_validate(BASE_CONFIG))

    with pytest.raises(OpenbookNotConfiguredError, match="openbook"):
        kit.harness.answer(AskRequest.from_question(QUESTION, sources=["a.md"]))


def test_explicit_openbook_and_stuffed_modes_report_no_auto_reason(tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())
    kit.client.script("mid-model", ok_result("stuffed"))

    pipeline = kit.harness.answer(AskRequest.from_question(QUESTION, mode="openbook", sources=[guide_path]))
    stuffed = kit.harness.answer(
        AskRequest.from_question(QUESTION, mode="stuffed", sources=[guide_path], tier="mid")
    )

    assert (pipeline.mode, pipeline.auto_reason) == ("openbook", None)
    assert (stuffed.mode, stuffed.auto_reason) == ("stuffed", None)


# --- CLI ---


def run_cli(monkeypatch, kit: HarnessKit, arguments: list[str]) -> int:
    monkeypatch.setattr(cli, "build_harness", lambda *args, **kwargs: kit.harness)
    return cli.main(arguments)


def test_cli_default_is_auto_and_the_trace_shows_the_choice(monkeypatch, capsys):
    kit = HarnessKit(make_config())
    kit.client.script("mid-model", ok_result("a nonce is used once"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "a nonce is used once"
    assert "mode          route (auto: no sources, one closed-book call)" in captured.err
    assert "route         mid (reason: auto)" in captured.err
    assert "triage        skipped" in captured.err


def test_cli_route_flag_triages(monkeypatch, capsys):
    kit = HarnessKit(make_config())
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("local-model", ok_result("routed"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?", "--route"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert kit.client.models_called() == ["triage-model", "local-model"]
    assert "mode          route\n" in captured.err
    assert "auto:" not in captured.err


@pytest.mark.parametrize("extra", [["--triage-tier", "triage"], ["--simulate", "triage-timeout"]])
def test_cli_triage_flags_imply_route_mode(monkeypatch, capsys, extra):
    kit = HarnessKit(make_config())
    kit.client.script("triage-model", triage_json("local", 0.95))
    kit.client.script("local-model", ok_result("routed"))
    kit.client.script("mid-model", ok_result("after the triage timeout"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?", *extra])

    assert exit_code == 0
    assert kit.run_record.auto_reason is None
    assert kit.run_record.route_reason in ("triage_label", "triage_timeout")


def test_cli_tier_flag_alone_is_one_forced_call(monkeypatch, capsys):
    kit = HarnessKit(make_config())
    kit.client.script("local-model", ok_result("forced"))

    exit_code = run_cli(monkeypatch, kit, ["ask", "what is a nonce?", "--tier", "local"])

    assert exit_code == 0
    assert kit.client.models_called() == ["local-model"]
    assert kit.run_record.route_reason == "forced"


def test_cli_sources_go_stuffed_when_they_fit(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", ok_result("Below 0.80."))

    exit_code = run_cli(monkeypatch, kit, ["ask", QUESTION, "--sources", guide_path])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "Below 0.80."
    assert "mode          stuffed (auto: sources fit, " in captured.err


def test_cli_require_citations_runs_the_pipeline(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())
    kit.client.script("ground-model", grounded_report())

    exit_code = run_cli(
        monkeypatch,
        kit,
        ["ask", QUESTION, "--sources", guide_path, "--require-citations", "--verifier-tier", "ground_chat"],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mode          openbook (auto: citations required)" in captured.err
    assert "Citations:" in captured.out


def test_cli_pipeline_only_simulation_selects_the_pipeline(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))
    kit.client.script("mid-model", cited_draft())

    exit_code = run_cli(
        monkeypatch, kit, ["ask", QUESTION, "--sources", guide_path, "--simulate", "grounding-error"]
    )

    assert exit_code == 0
    assert kit.run_record.mode == "openbook"
    assert kit.run_record.auto_reason is None


def test_cli_usage_errors_for_the_new_flags(monkeypatch, capsys, tmp_path, guide_path):
    kit = HarnessKit(windowed_config(tmp_path, 10_000))

    assert run_cli(monkeypatch, kit, ["ask", QUESTION, "--require-citations"]) == 2
    assert "nothing to cite" in capsys.readouterr().err

    assert run_cli(monkeypatch, kit, ["ask", QUESTION, "--route", "--sources", guide_path]) == 2
    assert "--route cannot be combined" in capsys.readouterr().err

    assert run_cli(monkeypatch, kit, ["ask", QUESTION, "--route", "--chain"]) == 2
    assert "--route cannot be combined" in capsys.readouterr().err

    assert (
        run_cli(monkeypatch, kit, ["ask", QUESTION, "--sources", guide_path, "--stuffed", "--require-citations"])
        == 2
    )
    assert "require_citations needs mode" in capsys.readouterr().err
