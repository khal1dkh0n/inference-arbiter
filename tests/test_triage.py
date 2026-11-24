from __future__ import annotations

import pytest

from inference_arbiter.schemas import Message, TriageDecision
from inference_arbiter.triage import (
    TRUNCATION_MARKER,
    build_triage_text,
    extract_text,
    run_triage,
    truncate_head_tail,
)

from conftest import FakeClient, error_result, make_config, ok_result, triage_json

IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAASECRETIMAGE"}}
FILE_PART = {"type": "file", "file": {"filename": "big.pdf", "file_data": "JVBERi0BULKYDOC"}}


def user_message(content) -> Message:
    return Message(role="user", content=content)


def test_extract_text_drops_images_and_attachments():
    content = [{"type": "text", "text": "what is in this picture?"}, IMAGE_PART, FILE_PART]
    assert extract_text(content) == "what is in this picture?"


def test_triage_uses_only_the_last_user_message_text():
    messages = [
        Message(role="system", content="system preamble"),
        user_message("an earlier question"),
        Message(role="assistant", content="an earlier answer"),
        user_message([IMAGE_PART, {"type": "text", "text": "the latest question"}]),
    ]
    assert build_triage_text(messages, max_chars=200) == "the latest question"


def test_truncation_keeps_head_and_tail():
    long_text = "H" * 500 + "T" * 500
    truncated = truncate_head_tail(long_text, max_chars=100)
    assert truncated == "H" * 50 + TRUNCATION_MARKER + "T" * 50
    assert truncate_head_tail("short", max_chars=100) == "short"


def run_with(result, messages=None):
    config = make_config()
    client = FakeClient()
    client.script("triage-model", result)
    outcome = run_triage(config, client, messages or [user_message("hello")])
    return outcome, client


def test_triage_call_is_text_only_deterministic_and_schema_constrained():
    messages = [user_message([{"type": "text", "text": "Q" * 1000}, IMAGE_PART, FILE_PART])]
    outcome, client = run_with(triage_json("local", 0.9), messages)

    assert outcome.status == "ok"
    assert len(client.calls) == 1
    triage_call = client.calls[0]
    assert triage_call.temperature == 0
    assert triage_call.think is False
    assert triage_call.max_output_tokens == 200
    assert triage_call.num_ctx == 4096
    assert triage_call.json_schema == TriageDecision.model_json_schema()
    sent_user_text = triage_call.messages[-1]["content"]
    assert isinstance(sent_user_text, str)
    assert "SECRETIMAGE" not in sent_user_text
    assert "BULKYDOC" not in sent_user_text
    assert len(sent_user_text) == 200 + len(TRUNCATION_MARKER)


def test_valid_output_becomes_a_decision():
    outcome, _ = run_with(triage_json("code", 0.93, "asks for SQL"))
    assert outcome.decision == TriageDecision(label="code", confidence=0.93, reason="asks for SQL")


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    [("timeout", "timeout"), ("connection", "error"), ("http_5xx", "error"), ("empty", "error")],
)
def test_client_failures_fail_closed_without_retry(error_type, expected_status):
    outcome, client = run_with(error_result(error_type))
    assert outcome.status == expected_status
    assert outcome.decision is None
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "raw_output",
    [
        "not json at all",
        '{"label": "local", "confidence": 0.9',
        '{"label": "local", "confidence": 1.7, "reason": "out of range"}',
        '{"label": "local_heavy", "confidence": 0.9, "reason": "never auto-routed"}',
        '{"label": "banana", "confidence": 0.9, "reason": "unknown label"}',
        '{"label": "local", "reason": "no confidence"}',
    ],
)
def test_invalid_output_is_invalid_with_no_guessed_label(raw_output):
    outcome, client = run_with(ok_result(raw_output))
    assert outcome.status == "invalid"
    assert outcome.decision is None
    assert len(client.calls) == 1
