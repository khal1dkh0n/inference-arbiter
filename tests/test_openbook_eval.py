"""Open-book eval tests: corpus manifest, question set, runner and scorecard. No network."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import httpx
import pytest
import yaml

from inference_arbiter.grading import Grade

from conftest import REAL_CONFIG_PATH, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fetch_corpus  # noqa: E402
import run_eval  # noqa: E402

GUIDE_TEXT = "# Guide\n\nEach list takes up to 1,024 patterns and 65,536 characters total per list.\n"
NOTES_TEXT = "# Notes\n\nThe router is available only in the global data region.\n"
GUIDE_URL = "https://docs.example.test/guide.md"
EVIDENCE_QUOTE = "65,536 characters total per list."

CASES = [
    {
        "id": "OB-A1",
        "question": "total characters per list?",
        "sources": ["guide"],
        "answerable": True,
        "expected_answer": "65536",
        "match": {"type": "numeric", "value": 65536, "tolerance": 0},
        "evidence": [{"document": "guide", "quote": EVIDENCE_QUOTE}],
        "category": "exact_value",
    },
    {
        "id": "OB-A2",
        "question": "which data region?",
        "sources": ["guide", "notes"],
        "answerable": True,
        "expected_answer": "global",
        "match": {"type": "contains", "all_of": ["global"]},
        "evidence": [{"document": "notes", "quote": "available only in the global data region."}],
        "category": "lookup",
    },
    {
        "id": "OB-U1",
        "question": "what does it cost?",
        "sources": ["guide"],
        "answerable": False,
        "expected_answer": "The sources do not say.",
        "match": {"type": "abstain"},
        "evidence": [],
        "category": "unanswerable",
    },
]
QUESTION_TO_ID = {case["question"]: case["id"] for case in CASES}


# ---- corpus manifest ----


def write_corpus(tmp_path, pin: bool = True):
    """A two-document corpus under tmp_path with its manifest. Returns (manifest path, root)."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "guide.md").write_text(GUIDE_TEXT, encoding="utf-8")
    (tmp_path / "notes.md").write_text(NOTES_TEXT, encoding="utf-8")
    documents = [
        {"id": "guide", "path": "data/guide.md", "url": GUIDE_URL},
        {"id": "notes", "path": "notes.md"},
    ]
    if pin:
        documents[0]["sha256"] = fetch_corpus.sha256_of_bytes(GUIDE_TEXT.encode("utf-8"))
        documents[1]["sha256"] = fetch_corpus.sha256_of_bytes(NOTES_TEXT.encode("utf-8"))
    manifest_path = tmp_path / "corpus.yaml"
    manifest_path.write_text(yaml.safe_dump({"version": 1, "documents": documents}), encoding="utf-8")
    return manifest_path, tmp_path


def test_verify_corpus_returns_text_for_every_document(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    corpus = fetch_corpus.verify_corpus(manifest_path, root)
    assert set(corpus) == {"guide", "notes"}
    assert corpus["guide"]["text"] == GUIDE_TEXT
    assert corpus["guide"]["url"] == GUIDE_URL
    assert corpus["notes"]["path"] == str(root / "notes.md")


def test_verify_corpus_refuses_a_missing_file(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    (root / "data" / "guide.md").unlink()
    with pytest.raises(fetch_corpus.CorpusError) as refusal:
        fetch_corpus.verify_corpus(manifest_path, root)
    message = str(refusal.value)
    assert "guide (data/guide.md): file is missing" in message
    assert "fetch_corpus.py" in message
    assert "notes" not in message.split("Run `")[0]


def test_verify_corpus_refuses_changed_content(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    (root / "notes.md").write_text(NOTES_TEXT + "one more line\n", encoding="utf-8")
    with pytest.raises(fetch_corpus.CorpusError) as refusal:
        fetch_corpus.verify_corpus(manifest_path, root)
    assert "notes (notes.md): content differs from the hash in the manifest" in str(refusal.value)


def test_verify_corpus_refuses_an_unpinned_document(tmp_path):
    manifest_path, root = write_corpus(tmp_path, pin=False)
    with pytest.raises(fetch_corpus.CorpusError) as refusal:
        fetch_corpus.verify_corpus(manifest_path, root)
    assert "manifest has no sha256" in str(refusal.value)


def test_verify_corpus_names_every_problem_at_once(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    (root / "data" / "guide.md").unlink()
    (root / "notes.md").write_text("changed", encoding="utf-8")
    problems = fetch_corpus.find_problems(fetch_corpus.load_manifest(manifest_path), root)
    assert [(problem["id"], problem["problem"]) for problem in problems] == [
        ("guide", "missing"),
        ("notes", "hash_mismatch"),
    ]


def test_load_manifest_rejects_duplicates_and_missing_fields(tmp_path):
    manifest_path = tmp_path / "corpus.yaml"
    manifest_path.write_text(yaml.safe_dump({"documents": [{"id": "a", "path": "x"}, {"id": "a", "path": "y"}]}))
    with pytest.raises(fetch_corpus.CorpusError, match="duplicate"):
        fetch_corpus.load_manifest(manifest_path)
    manifest_path.write_text(yaml.safe_dump({"documents": [{"id": "a"}]}))
    with pytest.raises(fetch_corpus.CorpusError, match="id and a path"):
        fetch_corpus.load_manifest(manifest_path)
    manifest_path.write_text(yaml.safe_dump({"version": 1}))
    with pytest.raises(fetch_corpus.CorpusError, match="documents"):
        fetch_corpus.load_manifest(manifest_path)


def test_corpus_fingerprint_changes_with_any_document_hash(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    corpus = fetch_corpus.verify_corpus(manifest_path, root)
    changed = {**corpus, "notes": {**corpus["notes"], "sha256": "0" * 64}}
    assert fetch_corpus.corpus_fingerprint(corpus) == fetch_corpus.corpus_fingerprint(dict(reversed(corpus.items())))
    assert fetch_corpus.corpus_fingerprint(corpus) != fetch_corpus.corpus_fingerprint(changed)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": fetch_corpus.USER_AGENT})


def test_fetch_downloads_only_missing_remote_documents_with_a_generic_client_name(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    (root / "data" / "guide.md").unlink()
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, content=GUIDE_TEXT.encode("utf-8"))

    with mock_client(handler) as client:
        log_lines = fetch_corpus.fetch_documents(fetch_corpus.load_manifest(manifest_path), root, client)

    assert [str(request.url) for request in seen_requests] == [GUIDE_URL]
    assert seen_requests[0].headers["user-agent"] == "corpus-fetch/1.0 (python-httpx)"
    assert "authorization" not in seen_requests[0].headers
    assert seen_requests[0].url.query == b""
    assert (root / "data" / "guide.md").read_text(encoding="utf-8") == GUIDE_TEXT
    assert log_lines[0].startswith("fetched  guide")
    # A second pass keeps the file; the repo document has no url and is never requested.
    with mock_client(handler) as client:
        log_lines = fetch_corpus.fetch_documents(fetch_corpus.load_manifest(manifest_path), root, client)
    assert len(seen_requests) == 1
    assert log_lines[0].startswith("kept")


def test_fetch_failure_is_logged_and_leaves_the_file_missing(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    (root / "data" / "guide.md").unlink()

    with mock_client(lambda request: httpx.Response(503)) as client:
        log_lines = fetch_corpus.fetch_documents(fetch_corpus.load_manifest(manifest_path), root, client)

    assert log_lines[0].startswith("FAILED   guide")
    assert not (root / "data" / "guide.md").exists()


def test_fetched_content_that_differs_from_the_manifest_is_refused_until_pinned(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    new_text = GUIDE_TEXT + "upstream edit\n"

    with mock_client(lambda request: httpx.Response(200, content=new_text.encode("utf-8"))) as client:
        fetch_corpus.fetch_documents(fetch_corpus.load_manifest(manifest_path), root, client, refetch=True)
    with pytest.raises(fetch_corpus.CorpusError, match="content differs"):
        fetch_corpus.verify_corpus(manifest_path, root)

    assert fetch_corpus.pin_manifest(manifest_path, root) == ["guide"]
    assert fetch_corpus.verify_corpus(manifest_path, root)["guide"]["text"] == new_text
    assert fetch_corpus.pin_manifest(manifest_path, root) == []
    assert manifest_path.read_text(encoding="utf-8").startswith("# Open-book eval corpus manifest")


def test_fetch_corpus_main_check_mode_reports_and_uses_no_network(tmp_path, capsys, monkeypatch):
    manifest_path, root = write_corpus(tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("--check must not open a client")

    monkeypatch.setattr(fetch_corpus.httpx, "Client", no_network)
    assert fetch_corpus.main(["--check", "--manifest", str(manifest_path), "--root", str(root)]) == 0
    (root / "notes.md").write_text("changed", encoding="utf-8")
    assert fetch_corpus.main(["--check", "--manifest", str(manifest_path), "--root", str(root)]) == 1
    assert "content differs" in capsys.readouterr().err


# ---- question set against the corpus ----


def test_validate_openbook_cases_accepts_a_fitting_set(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    corpus = fetch_corpus.verify_corpus(manifest_path, root)
    assert run_eval.validate_openbook_cases(CASES, corpus) == []


def test_validate_openbook_cases_reports_each_misfit(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    corpus = fetch_corpus.verify_corpus(manifest_path, root)
    bad_cases = [
        {**CASES[0], "id": "B1", "sources": ["guide", "nowhere"]},
        {**CASES[0], "id": "B2", "evidence": [{"document": "guide", "quote": "65536 characters in total"}]},
        {**CASES[0], "id": "B3", "evidence": [{"document": "notes", "quote": "global data region."}]},
        {**CASES[0], "id": "B4", "evidence": []},
        {**CASES[2], "id": "B5", "evidence": [{"document": "guide", "quote": EVIDENCE_QUOTE}]},
        {**CASES[0], "id": "B6", "sources": []},
    ]
    problems = run_eval.validate_openbook_cases(bad_cases, corpus)
    assert "B1: source 'nowhere' is not in the corpus manifest" in problems
    assert "B2: evidence quote not found verbatim in 'guide'" in problems
    assert "B3: evidence document 'notes' is not among its sources" in problems
    assert "B4: answerable case without an evidence quote" in problems
    assert "B5: unanswerable case must not have evidence" in problems
    assert "B6: no sources" in problems


def test_evidence_quote_may_span_a_line_break_in_the_source(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    corpus = fetch_corpus.verify_corpus(manifest_path, root)
    corpus["guide"]["text"] = "Each list takes 65,536 characters\ntotal per list.\n"
    assert run_eval.validate_openbook_cases([CASES[0]], corpus) == []


def write_questions(tmp_path, cases=CASES):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(yaml.safe_dump({"version": 1, "cases": cases}), encoding="utf-8")
    return questions_path


def test_load_openbook_ties_the_result_to_questions_and_corpus(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    cases, golden_info, corpus = run_eval.load_openbook(write_questions(tmp_path), manifest_path, root)
    assert [case["id"] for case in cases] == ["OB-A1", "OB-A2", "OB-U1"]
    assert golden_info["corpus_fingerprint"] == fetch_corpus.corpus_fingerprint(corpus)
    assert run_eval.same_question_set(dict(golden_info), golden_info)
    assert not run_eval.same_question_set({**golden_info, "corpus_fingerprint": "other"}, golden_info)
    assert not run_eval.same_question_set({"sha256": golden_info["sha256"]}, golden_info)


def test_load_openbook_refuses_questions_that_do_not_fit(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    bad_case = {**CASES[0], "evidence": [{"document": "guide", "quote": "not in the file"}]}
    with pytest.raises(run_eval.QuestionSetError, match="evidence quote not found"):
        run_eval.load_openbook(write_questions(tmp_path, [bad_case]), manifest_path, root)


def test_sources_all_supplies_every_document_of_the_manifest(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    whole_corpus_case = {**CASES[0], "id": "OB-ALL", "sources": "all"}
    cases, _, corpus = run_eval.load_openbook(write_questions(tmp_path, [whole_corpus_case]), manifest_path, root)
    assert cases[0]["sources"] == list(corpus)
    assert len(cases[0]["sources"]) > 1


def test_repo_large_question_set_fits_the_large_corpus():
    """The committed large-corpus questions against the fetched pages, when they are present."""
    manifest_path = REPO_ROOT / "eval/openbook/corpus_large.yaml"
    if fetch_corpus.find_problems(fetch_corpus.load_manifest(manifest_path), REPO_ROOT):
        pytest.skip("large corpus not fetched or differs; run scripts/fetch_corpus.py --manifest")
    cases, _, corpus = run_eval.load_openbook(REPO_ROOT / "eval/openbook/questions_large.yaml", manifest_path)
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["sources"] == list(corpus) for case in cases)
    assert sum(1 for case in cases if not case["answerable"]) >= 3


def test_same_question_set_still_accepts_closed_book_files():
    assert run_eval.same_question_set({"sha256": "abc"}, {"sha256": "abc", "path": "golden.yaml"})
    assert not run_eval.same_question_set({"sha256": "abc"}, {"sha256": "def"})


def test_repo_question_set_fits_the_repo_corpus():
    """The committed questions against the real corpus, when it has been fetched."""
    manifest = fetch_corpus.load_manifest(REPO_ROOT / run_eval.DEFAULT_CORPUS_MANIFEST_PATH)
    if fetch_corpus.find_problems(manifest, REPO_ROOT):
        pytest.skip("corpus not fetched or differs; run scripts/fetch_corpus.py")
    cases, golden_info, corpus = run_eval.load_openbook(
        REPO_ROOT / run_eval.DEFAULT_OPENBOOK_QUESTIONS_PATH, REPO_ROOT / run_eval.DEFAULT_CORPUS_MANIFEST_PATH
    )
    assert len({case["id"] for case in cases}) == len(cases)
    unanswerable = [case for case in cases if not case["answerable"]]
    assert 4 <= len(unanswerable) <= 5
    assert all(case["match"]["type"] == "abstain" for case in unanswerable)
    assert all(case["match"]["type"] in ("exact", "contains", "numeric", "rubric")
               for case in cases if case["answerable"])
    assert any(len(case["sources"]) > 1 for case in cases)
    assert any(len(case["sources"]) == 1 for case in cases)


def test_repo_manifest_lists_only_ignored_paths_for_fetched_documents():
    manifest = fetch_corpus.load_manifest(REPO_ROOT / run_eval.DEFAULT_CORPUS_MANIFEST_PATH)
    for document in manifest["documents"]:
        assert document.get("sha256"), document["id"]
        if document.get("url"):
            assert document["path"].startswith("data/corpus/"), document["id"]
        else:
            assert (REPO_ROOT / document["path"]).is_file(), document["id"]


# ---- strategies and requests ----


class FakeAskRequest:
    """Stands in for AskRequest so the tests do not depend on which modes the harness has."""

    model_fields = {"mode": None, "sources": None, "docs_root": None}

    @classmethod
    def from_question(cls, question, **overrides):
        extra = {"verifier_tier": overrides["verifier_tier"]} if "verifier_tier" in overrides else {}
        if "docs_root" in overrides:
            extra["docs_root"] = overrides["docs_root"]
        return SimpleNamespace(question=question, tier=overrides.get("tier"), mode=overrides.get("mode", "route"),
                               sources=overrides.get("sources"), **extra)


@pytest.fixture
def corpus(tmp_path):
    manifest_path, root = write_corpus(tmp_path)
    return fetch_corpus.verify_corpus(manifest_path, root)


@pytest.fixture
def modes(monkeypatch):
    """Set which open-book modes the fake harness accepts."""
    monkeypatch.setattr(run_eval, "AskRequest", FakeAskRequest)

    def set_supported(*mode_names):
        monkeypatch.setattr(run_eval, "harness_mode_supported", lambda mode_name: mode_name in mode_names)

    set_supported("openbook", "stuffed")
    return set_supported


def fake_result(answer, final_tier="mid", *, status="ok", fallback_used=False, cost_usd=0.0,
                latency_ms=100.0, **extra):
    fields = dict(
        trace_id="trace", status=status, answer=answer,
        error_type=None if status == "ok" else "http_5xx", error_message=None if status == "ok" else "boom",
        triage_label=None, triage_confidence=None, route_reason="forced",
        selected_tier=final_tier, final_tier=final_tier, final_model=f"{final_tier}-model",
        fallback_used=fallback_used, total_latency_ms=latency_ms, total_cost_usd=cost_usd, calls=[],
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


class FakeHarness:
    def __init__(self, responder):
        self._responder = responder
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return self._responder(request)


class FakeJudge:
    """Rubric: pass when the answer contains 'good'. Abstention: met when it contains 'do not contain'."""

    model = "judge-model"

    def __init__(self):
        self.calls = 0

    def judge(self, case, answer) -> Grade:
        self.calls += 1
        marker = "do not contain" if "declines to answer" in case["expected_answer"] else "good"
        return Grade(status="pass" if marker in answer else "fail", reason="judged", match_type="rubric",
                     judge_tier="mid", judge_model=self.model, judge_cost_usd=0.001)


def strategy(text):
    return run_eval.parse_strategy(text)


def test_parse_open_book_strategies():
    assert strategy("openbook") == run_eval.Strategy("openbook", "openbook", None)
    assert strategy("stuffed:mid") == run_eval.Strategy("stuffed:mid", "stuffed", "mid")
    assert strategy("closed:local") == run_eval.Strategy("closed:local", "closed", "local")
    for bad_text in ("stuffed:", "closed", "openbook:mid", "stuffed"):
        with pytest.raises(ValueError):
            run_eval.parse_strategy(bad_text)


def test_requests_per_strategy_when_the_harness_has_both_modes(corpus, modes):
    documents = run_eval.case_documents(CASES[1], corpus)
    paths = [corpus["guide"]["path"], corpus["notes"]["path"]]

    openbook_request, openbook_stuffing = run_eval.build_openbook_request(strategy("openbook"), CASES[1], documents)
    assert (openbook_request.mode, openbook_request.sources, openbook_request.tier) == ("openbook", paths, None)
    assert openbook_stuffing is None
    assert not hasattr(openbook_request, "verifier_tier")

    stuffed_request, stuffed_by = run_eval.build_openbook_request(strategy("stuffed:mid"), CASES[1], documents)
    assert (stuffed_request.mode, stuffed_request.sources, stuffed_request.tier) == ("stuffed", paths, "mid")
    assert stuffed_request.question == CASES[1]["question"]
    assert stuffed_by == "harness"

    closed_request, closed_stuffing = run_eval.build_openbook_request(strategy("closed:mid"), CASES[1], documents)
    assert (closed_request.mode, closed_request.sources, closed_request.tier) == ("route", None, "mid")
    assert closed_request.question == CASES[1]["question"]
    assert closed_stuffing is None


def test_verifier_tier_reaches_only_the_openbook_strategy(corpus, modes):
    documents = run_eval.case_documents(CASES[0], corpus)
    grounded, _ = run_eval.build_openbook_request(strategy("openbook"), CASES[0], documents, "ground_chat")
    stuffed, _ = run_eval.build_openbook_request(strategy("stuffed:mid"), CASES[0], documents, "ground_chat")
    closed, _ = run_eval.build_openbook_request(strategy("closed:mid"), CASES[0], documents, "ground_chat")
    assert grounded.verifier_tier == "ground_chat"
    assert not hasattr(stuffed, "verifier_tier") and not hasattr(closed, "verifier_tier")


def test_stuffed_is_built_by_the_runner_when_the_harness_has_no_stuffed_mode(corpus, modes):
    modes()
    documents = run_eval.case_documents(CASES[1], corpus)
    request, stuffed_by = run_eval.build_openbook_request(strategy("stuffed:mid"), CASES[1], documents)
    assert stuffed_by == "runner"
    assert (request.mode, request.sources, request.tier) == ("route", None, "mid")
    assert GUIDE_TEXT in request.question and NOTES_TEXT in request.question
    assert request.question.index(GUIDE_TEXT) < request.question.index("Question: which data region?")
    assert request.question.endswith("Question: which data region?")
    assert '<source id="notes">' in request.question


def test_harness_mode_supported_reads_the_real_request_schema():
    assert run_eval.harness_mode_supported("route") == ("sources" in run_eval.AskRequest.model_fields)
    assert run_eval.harness_mode_supported("no-such-mode") is False


def test_openbook_reported_not_runnable_until_the_harness_accepts_the_mode(corpus, modes):
    modes()
    harness = FakeHarness(lambda request: fake_result("65536"))
    run = run_eval.run_strategies(
        CASES, [strategy("openbook"), strategy("closed:mid")], lambda _: harness, FakeJudge(),
        case_runner=lambda *arguments: run_eval.run_openbook_case(*arguments, corpus),
    )
    assert "openbook" in run["not_runnable"]
    assert "mode 'openbook'" in run["not_runnable"]["openbook"]
    assert {row["strategy"] for row in run["records"]} == {"closed:mid"}


# ---- running and grading one case ----


def run_one(case, corpus, responder, strategy_text="openbook", judge=None):
    harness = FakeHarness(responder)
    record = run_eval.run_openbook_case(harness, strategy(strategy_text), case, judge or FakeJudge(), corpus)
    return record, harness


def good_citation(corpus, quote=EVIDENCE_QUOTE, document_id="guide"):
    return {"source": corpus[document_id]["path"], "passage_id": "p3", "quote": quote}


def test_grounded_answer_passes_every_column(corpus, modes):
    record, harness = run_one(
        CASES[0], corpus,
        lambda request: fake_result("65536", citations=[good_citation(corpus)], grounding_status="grounded",
                                    abstained=False, passages_retrieved=4, repaired=True, mode="openbook"),
    )
    assert harness.requests[0].sources == [corpus["guide"]["path"]]
    assert record["outcome"] == "pass"
    assert (record["answer_grade"], record["abstention_grade"]) == ("pass", "pass")
    assert (record["citation_validity"], record["evidence_hit"]) == ("pass", "pass")
    assert record["abstained"] is False
    assert (record["grounding_status"], record["passages_retrieved"], record["repaired"]) == ("grounded", 4, True)
    assert record["citations"] == [good_citation(corpus)]
    assert record["suite"] == "openbook" and record["answerable"] is True


def test_right_answer_with_an_invented_quote_is_right_but_not_validly_cited(corpus, modes):
    invented = good_citation(corpus, quote="the total is 65,536 characters")
    record, _ = run_one(CASES[0], corpus, lambda request: fake_result("65536", citations=[invented], abstained=False))
    assert record["outcome"] == "pass"
    assert (record["citation_validity"], record["evidence_hit"]) == ("fail", "fail")
    assert record["citation_validity_detail"]["invalid"][0]["problem"] == "quote not found in the cited source"


def test_valid_citation_away_from_the_evidence_misses_the_evidence(corpus, modes):
    elsewhere = good_citation(corpus, quote="Each list takes up to 1,024 patterns")
    record, _ = run_one(CASES[0], corpus, lambda request: fake_result("65536", citations=[elsewhere], abstained=False))
    assert (record["citation_validity"], record["evidence_hit"]) == ("pass", "fail")


def test_pipeline_abstaining_on_an_unanswerable_case_passes_without_the_judge(corpus, modes):
    judge = FakeJudge()
    record, _ = run_one(
        CASES[2], corpus,
        lambda request: fake_result(None, abstained=True, citations=[], grounding_status="ungrounded"),
        judge=judge,
    )
    assert record["outcome"] == "pass"
    assert (record["answer_grade"], record["abstention_grade"]) == ("not_applicable", "pass")
    assert record["abstention_decided_by"] == "abstained_flag"
    assert (record["citation_validity"], record["evidence_hit"]) == ("not_applicable", "not_applicable")
    assert judge.calls == 0


def test_pipeline_abstaining_on_an_answerable_case_fails(corpus, modes):
    record, _ = run_one(CASES[0], corpus, lambda request: fake_result("not found", abstained=True, citations=[]))
    assert record["outcome"] == "fail"
    assert (record["answer_grade"], record["abstention_grade"], record["abstained"]) == ("fail", "fail", True)
    assert record["citation_validity"] == "not_applicable"


def test_stuffed_answer_to_an_unanswerable_case_is_judged_from_its_text(corpus, modes):
    declined, _ = run_one(
        CASES[2], corpus, lambda request: fake_result("The sources do not contain a price.", final_tier="mid"),
        strategy_text="stuffed:mid",
    )
    assert (declined["outcome"], declined["abstained"], declined["abstention_decided_by"]) == ("pass", True, "judge")
    assert declined["judge_cost_usd"] == 0.001
    assert declined["self_graded"] is False

    answered, _ = run_one(
        CASES[2], corpus, lambda request: fake_result("It costs 0.01 USD.", final_tier="mid"),
        strategy_text="stuffed:mid",
    )
    assert (answered["outcome"], answered["abstention_grade"], answered["abstained"]) == ("fail", "fail", False)
    assert answered["stuffed_by"] == "harness"


def test_strategies_that_do_not_cite_get_not_applicable_citation_columns(corpus, modes):
    for strategy_text in ("stuffed:mid", "closed:mid"):
        record, _ = run_one(CASES[0], corpus, lambda request: fake_result("65536"), strategy_text=strategy_text)
        assert record["outcome"] == "pass"
        assert (record["citation_validity"], record["evidence_hit"]) == ("not_applicable", "not_applicable")
        assert record["abstained_flag"] is None


def test_closed_strategy_sends_no_sources(corpus, modes):
    record, harness = run_one(CASES[0], corpus, lambda request: fake_result("1024"), strategy_text="closed:mid")
    assert harness.requests[0].sources is None
    assert harness.requests[0].question == CASES[0]["question"]
    assert (record["outcome"], record["abstained"]) == ("fail", False)


def test_forced_tier_run_answered_by_another_tier_is_excluded(corpus, modes):
    record, _ = run_one(
        CASES[0], corpus, lambda request: fake_result("65536", final_tier="local", fallback_used=True),
        strategy_text="stuffed:mid",
    )
    assert record["outcome"] == "fallback_excluded"
    assert record["answer_grade"] is None


def test_failed_run_and_raising_harness_are_errors(corpus, modes):
    failed, _ = run_one(CASES[0], corpus, lambda request: fake_result(None, status="error"))
    assert (failed["outcome"], failed["answer_grade"]) == ("error", None)

    def raising(request):
        raise RuntimeError("pipeline crashed")

    crashed, _ = run_one(CASES[0], corpus, raising)
    assert crashed["outcome"] == "error"
    assert "pipeline crashed" in crashed["error_message"]


def test_no_answer_and_no_abstention_is_an_error_not_a_grade(corpus, modes):
    record, _ = run_one(CASES[2], corpus, lambda request: fake_result(None, abstained=False))
    assert record["outcome"] == "error"


def test_self_graded_when_the_judge_model_wrote_the_answer(corpus, modes):
    record, _ = run_one(
        CASES[2], corpus, lambda request: fake_result("It costs 0.01 USD.", final_tier="judge"),
        strategy_text="stuffed:judge",
    )
    assert record["self_graded"] is True


def test_regrade_uses_stored_citations_and_flag_without_running_a_model(corpus, modes):
    record, _ = run_one(
        CASES[0], corpus,
        lambda request: fake_result("65536", citations=[good_citation(corpus)], abstained=False),
    )
    stale = {**record, "outcome": "fail", "answer_grade": "fail", "citation_validity": "fail", "evidence_hit": "fail"}
    error_record = {**record, "outcome": "error", "answer": None, "answer_grade": None}

    regraded = run_eval.regrade_openbook_records([stale, error_record], CASES, corpus, FakeJudge())

    assert (regraded[0]["outcome"], regraded[0]["citation_validity"], regraded[0]["evidence_hit"]) == (
        "pass", "pass", "pass",
    )
    assert regraded[1]["outcome"] == "error"
    assert stale["outcome"] == "fail"


# ---- scorecard arithmetic ----


def scored(strategy_name, case_id, *, answerable=True, outcome=None, answer_grade=None, abstention_grade="pass",
           abstained=False, citation_validity="not_applicable", evidence_hit="not_applicable", repaired=False,
           cost_usd=0.0, latency_ms=100.0, category="lookup", **extra):
    if outcome is None:
        outcome = answer_grade if answerable else abstention_grade
    fields = dict(
        strategy=strategy_name, case_id=case_id, category=category, outcome=outcome, answerable=answerable,
        answer_grade=answer_grade if answerable else "not_applicable", abstention_grade=abstention_grade,
        abstained=abstained, citation_validity=citation_validity, evidence_hit=evidence_hit, repaired=repaired,
        cost_usd=cost_usd, judge_cost_usd=0.0, latency_ms=latency_ms, fallback_used=False, self_graded=False,
        stuffed_by=None,
    )
    fields.update(extra)
    return fields


def openbook_records():
    return [
        scored("openbook", "A1", answer_grade="pass", citation_validity="pass", evidence_hit="pass",
               cost_usd=0.01, latency_ms=1000.0),
        scored("openbook", "A2", answer_grade="pass", citation_validity="fail", evidence_hit="fail",
               repaired=True, cost_usd=0.03, latency_ms=3000.0),
        scored("openbook", "A3", answer_grade="fail", abstention_grade="fail", abstained=True,
               latency_ms=2000.0),
        scored("openbook", "A4", answer_grade="fail", citation_validity="pass", evidence_hit="fail",
               latency_ms=4000.0),
        scored("openbook", "A5", answer_grade="ungraded", citation_validity="pass", evidence_hit="pass",
               latency_ms=5000.0),
        scored("openbook", "A6", outcome="error", answer_grade=None, abstention_grade=None, abstained=None,
               citation_validity=None, evidence_hit=None, latency_ms=9000.0),
        scored("openbook", "U1", answerable=False, abstention_grade="pass", abstained=True,
               category="unanswerable", latency_ms=500.0),
        scored("openbook", "U2", answerable=False, abstention_grade="fail", abstained=False,
               citation_validity="pass", category="unanswerable", repaired=True, latency_ms=6000.0),
        scored("openbook", "U3", answerable=False, abstention_grade="ungraded", abstained=None,
               category="unanswerable", latency_ms=700.0),
    ]


def test_summarise_openbook_strategy_keeps_every_measure_separate():
    summary = run_eval.summarise_openbook_strategy("openbook", openbook_records())

    assert (summary["cases"], summary["answerable_cases"], summary["unanswerable_cases"]) == (9, 6, 3)
    # Answer accuracy: answerable cases only; ungraded and error are left out.
    assert summary["answer"] == {"pass": 2, "fail": 2, "graded": 4, "rate": 0.5}
    assert summary["answer_accuracy"] == 0.5
    # Abstention accuracy: unanswerable cases only.
    assert summary["abstention"] == {"pass": 1, "fail": 1, "graded": 2, "rate": 0.5}
    assert summary["false_abstentions"] == 1
    assert summary["hallucinated_answers"] == 1
    # Citation validity: every run that had citations to check, answerable or not.
    assert summary["citation_validity"] == {"pass": 4, "fail": 1, "graded": 5, "rate": 0.8}
    # Evidence hit: answerable runs only.
    assert summary["evidence_hit"] == {"pass": 2, "fail": 2, "graded": 4, "rate": 0.5}
    assert (
        summary["escalations_attempted"], summary["escalations_succeeded"], summary["escalations_failed"],
    ) == (2, 2, 0)
    assert (summary["ungraded"], summary["error"], summary["fallback_excluded"]) == (2, 1, 0)
    assert summary["total_cost_usd"] == 0.04
    # Latency: the error run (9000 ms) is left out; 8 values, nearest rank.
    assert summary["latency_p50_ms"] == 2000.0
    assert summary["latency_p95_ms"] == 6000.0
    assert summary["by_category"]["unanswerable"] == {"cases": 3, "pass": 1, "graded": 2, "accuracy": 0.5}


def test_non_citing_strategy_has_no_citation_rate():
    records = [
        scored("stuffed:mid", "A1", answer_grade="pass", stuffed_by="runner"),
        scored("stuffed:mid", "U1", answerable=False, abstention_grade="fail", abstained=False),
    ]
    summary = run_eval.summarise_openbook_strategy("stuffed:mid", records)
    assert summary["citation_validity"] == {"pass": 0, "fail": 0, "graded": 0, "rate": None}
    assert summary["evidence_hit"]["rate"] is None
    assert summary["answer_accuracy"] == 1.0
    assert summary["abstention_accuracy"] == 0.0
    assert summary["stuffed_by"] == ["runner"]


def test_openbook_scorecard_compares_with_stuffed_mid_on_per_case_outcome():
    records = [
        *openbook_records(),
        scored("stuffed:mid", "A1", answer_grade="pass"),
        scored("stuffed:mid", "A2", answer_grade="fail"),
        scored("stuffed:mid", "A3", answer_grade="pass"),
        scored("stuffed:mid", "A4", answer_grade="fail"),
        scored("stuffed:mid", "A5", answer_grade="pass"),
        scored("stuffed:mid", "A6", answer_grade="pass"),
        scored("stuffed:mid", "U1", answerable=False, abstention_grade="fail", abstained=False),
        scored("stuffed:mid", "U2", answerable=False, abstention_grade="pass", abstained=True),
        scored("stuffed:mid", "U3", answerable=False, abstention_grade="pass", abstained=True),
        scored("closed:mid", "A1", answer_grade="fail"),
    ]
    scorecard = run_eval.build_openbook_scorecard(records)

    assert (scorecard["baseline"], scorecard["baseline_present"]) == ("stuffed:mid", True)
    assert [summary["strategy"] for summary in scorecard["strategies"]] == ["openbook", "stuffed:mid", "closed:mid"]
    comparison = {row["strategy"]: row for row in scorecard["head_to_head"]}["openbook"]
    assert (comparison["won"], comparison["won_cases"]) == (2, ["A2", "U1"])
    assert (comparison["lost"], comparison["lost_cases"]) == (2, ["A3", "U2"])
    assert (comparison["both_right"], comparison["both_wrong"]) == (1, 1)
    # A5 and U3 ungraded, A6 an error: three cases cannot be compared.
    assert comparison["not_comparable"] == 3
    assert comparison["cost_ratio"] is None


def test_openbook_scorecard_without_the_baseline_has_no_head_to_head():
    scorecard = run_eval.build_openbook_scorecard(openbook_records())
    assert scorecard["baseline_present"] is False
    assert scorecard["head_to_head"] == []


def test_print_openbook_scorecard_shows_separate_columns_and_caveats(capsys):
    records = [
        *openbook_records(),
        scored("stuffed:mid", "A1", answer_grade="pass", stuffed_by="runner"),
        scored("stuffed:mid", "U1", answerable=False, abstention_grade="pass", abstained=True, stuffed_by="runner"),
    ]
    run_eval.print_openbook_scorecard(run_eval.build_openbook_scorecard(records))
    printed = capsys.readouterr().out
    for column in ("answer acc", "abstain acc", "false abst", "halluc", "cite valid", "evid hit", "esc tried", "esc ok", "esc failed",
                   "cost usd", "p50 ms", "p95 ms"):
        assert column in printed
    assert "2/4 50%" in printed
    assert "4/5 80%" in printed
    assert "n/a" in printed
    assert "Head-to-head against stuffed:mid" in printed
    assert "Caveat: small sample. With 4 cases one case moves accuracy by 25 points" in printed
    assert "Abstention accuracy rests on 2 unanswerable cases: one case moves it by 50 points." in printed
    assert "stuffed:mid pasted the sources in this script" in printed


def test_print_openbook_scorecard_with_no_results(capsys):
    run_eval.print_openbook_scorecard(run_eval.build_openbook_scorecard([]))
    assert "No results to score." in capsys.readouterr().out


# ---- main: the eval refuses to run on an unknown corpus ----


@pytest.fixture
def main_environment(monkeypatch, tmp_path):
    """main() with no .env read, no clients and a harness that fails the test if it is built."""
    built: list[str] = []
    monkeypatch.setattr(run_eval, "load_env_file", lambda path: None)
    monkeypatch.setattr(run_eval, "build_clients", lambda config: {"ollama": object(), "openrouter": object()})

    def build_harness(**arguments):
        built.append("harness")
        return FakeHarness(lambda request: fake_result("65536"))

    monkeypatch.setattr(run_eval, "build_harness", build_harness)
    return built


def openbook_arguments(tmp_path, manifest_path, questions_path, *extra):
    return [
        "--suite", "openbook", "--config", str(REAL_CONFIG_PATH), "--corpus", str(manifest_path),
        "--golden", str(questions_path), "--out", str(tmp_path / "out.json"), "--db", str(tmp_path / "t.db"),
        "--routes-out", str(tmp_path / "routes.json"), *extra,
    ]


def absolute_manifest(tmp_path):
    """A manifest whose paths are absolute, so main() finds the files whatever its root is."""
    manifest_path, root = write_corpus(tmp_path)
    manifest = fetch_corpus.load_manifest(manifest_path)
    for document in manifest["documents"]:
        document["path"] = str(root / document["path"])
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest_path, root


def test_main_refuses_to_run_when_a_corpus_file_differs(tmp_path, capsys, main_environment):
    manifest_path, root = absolute_manifest(tmp_path)
    (root / "notes.md").write_text(NOTES_TEXT + "edited upstream\n", encoding="utf-8")

    exit_code = run_eval.main(
        openbook_arguments(tmp_path, manifest_path, write_questions(tmp_path), "--strategies", "closed:mid")
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "content differs from the hash in the manifest" in captured.err
    assert "notes" in captured.err
    assert main_environment == []
    assert not (tmp_path / "out.json").exists()


def test_main_refuses_to_run_when_a_corpus_file_is_missing(tmp_path, capsys, main_environment):
    manifest_path, root = absolute_manifest(tmp_path)
    (root / "data" / "guide.md").unlink()

    exit_code = run_eval.main(
        openbook_arguments(tmp_path, manifest_path, write_questions(tmp_path), "--strategies", "closed:mid")
    )

    assert exit_code == 2
    assert "file is missing" in capsys.readouterr().err
    assert main_environment == []


def test_main_rejects_strategies_from_the_other_suite(tmp_path, capsys, main_environment):
    manifest_path, _ = absolute_manifest(tmp_path)
    questions_path = write_questions(tmp_path)

    assert run_eval.main(openbook_arguments(tmp_path, manifest_path, questions_path, "--strategies", "tier:mid")) == 2
    assert "do not belong to the openbook suite" in capsys.readouterr().err
    assert run_eval.main(["--strategies", "stuffed:mid", "--config", str(REAL_CONFIG_PATH)]) == 2
    assert "do not belong to the closedbook suite" in capsys.readouterr().err


def test_main_runs_the_open_book_suite_and_writes_no_measured_routes(tmp_path, capsys, main_environment):
    manifest_path, _ = absolute_manifest(tmp_path)
    routes_path = tmp_path / "routes.json"
    routes_path.write_text("EXISTING ROUTES", encoding="utf-8")

    exit_code = run_eval.main(
        openbook_arguments(tmp_path, manifest_path, write_questions(tmp_path, CASES[:2]),
                           "--strategies", "closed:mid", "stuffed:mid")
    )

    printed = capsys.readouterr().out
    written = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert written["suite"] == "openbook"
    assert written["golden"]["corpus_fingerprint"]
    assert written["scorecard"]["baseline"] == "stuffed:mid"
    assert {row["strategy"] for row in written["records"]} == {"closed:mid", "stuffed:mid"}
    assert "Open-book scorecard" in printed
    assert routes_path.read_text(encoding="utf-8") == "EXISTING ROUTES"


# ---- the mcp strategy: the case's documents staged into a folder of their own ----


def test_parse_the_mcp_strategy():
    assert strategy("mcp") == run_eval.Strategy("mcp", "mcp", None)
    assert "mcp" in run_eval.SUITE_KINDS[run_eval.SUITE_OPENBOOK]
    assert "mcp" not in run_eval.SUITE_DEFAULT_STRATEGIES[run_eval.SUITE_OPENBOOK]
    with pytest.raises(ValueError):
        run_eval.parse_strategy("mcp:mid")


def test_stage_documents_copies_exactly_the_case_documents(corpus, tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    names = run_eval.stage_documents(run_eval.case_documents(CASES[1], corpus), staged)
    assert names == ["guide.md", "notes.md"]
    assert sorted(path.name for path in staged.iterdir()) == ["guide.md", "notes.md"]
    assert (staged / "guide.md").read_text(encoding="utf-8") == GUIDE_TEXT
    assert not any(path.is_symlink() for path in staged.iterdir())


def test_stage_documents_refuses_two_documents_with_one_file_name(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for folder in ("a", "b"):
        (tmp_path / folder / "same.md").write_text(folder, encoding="utf-8")
    documents = {"one": {"path": str(tmp_path / "a" / "same.md")}, "two": {"path": str(tmp_path / "b" / "same.md")}}
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(run_eval.QuestionSetError, match="same.md"):
        run_eval.stage_documents(documents, staged)


def test_mcp_case_is_asked_of_a_staged_folder_that_is_gone_afterwards(corpus, modes):
    modes("openbook", "stuffed", "mcp")
    seen = {}

    def responder(request):
        folder = run_eval.Path(request.docs_root)
        seen["folder"] = folder
        seen["files"] = {path.name: path.read_text(encoding="utf-8") for path in folder.iterdir()}
        return fake_result(
            "65536",
            citations=[{"source": "guide.md", "passage_id": "S1-P1", "quote": EVIDENCE_QUOTE}],
            grounding_status="grounded", abstained=False, passages_retrieved=1, mode="mcp",
            plan_status="partial", documents_listed=1, documents_read=1, tool_server_stopped=True,
            tool_calls=[
                SimpleNamespace(origin="code", status="ok"),
                SimpleNamespace(origin="plan", status="refused"),
                SimpleNamespace(origin="plan", status="ok"),
            ],
        )

    record, harness = run_one(CASES[0], corpus, responder, strategy_text="mcp")

    request = harness.requests[0]
    assert (request.mode, request.sources, request.tier) == ("mcp", None, None)
    assert seen["files"] == {"guide.md": GUIDE_TEXT}
    assert not seen["folder"].exists()
    assert record["outcome"] == "pass"
    assert (record["citation_validity"], record["evidence_hit"]) == ("pass", "pass")
    assert (record["plan_status"], record["documents_listed"], record["documents_read"]) == ("partial", 1, 1)
    assert (record["tool_calls_total"], record["tool_calls_refused"], record["tool_calls_from_plan"]) == (3, 1, 2)
    assert record["tool_server_stopped"] is True and record["stuffed_by"] is None


def test_mcp_answer_without_a_citation_fails_citation_validity(corpus, modes):
    modes("mcp")
    record, _ = run_one(
        CASES[0], corpus, lambda request: fake_result("65536", citations=[], abstained=False, mode="mcp"),
        strategy_text="mcp",
    )
    assert record["outcome"] == "pass" and record["citation_validity"] == "fail"


def test_mcp_verifier_tier_is_passed_on(corpus, modes):
    modes("mcp")
    request, stuffed_by = run_eval.build_openbook_request(
        strategy("mcp"), CASES[0], run_eval.case_documents(CASES[0], corpus), "ground_chat", docs_root="/staged"
    )
    assert (request.mode, request.docs_root, request.verifier_tier, stuffed_by) == ("mcp", "/staged", "ground_chat", None)


def test_mcp_reported_not_runnable_until_the_harness_accepts_the_mode(corpus, modes):
    modes("openbook")
    harness = FakeHarness(lambda request: fake_result("65536"))
    run = run_eval.run_strategies(
        CASES, [strategy("mcp"), strategy("closed:mid")], lambda _: harness, FakeJudge(),
        case_runner=lambda *arguments: run_eval.run_openbook_case(*arguments, corpus),
    )
    assert "mode 'mcp'" in run["not_runnable"]["mcp"]
    assert {row["strategy"] for row in run["records"]} == {"closed:mid"}


def test_mcp_strategy_runs_through_the_real_harness_gate_and_grader(corpus, tmp_path):
    """The whole path with only the models and the transport faked: staging, the server's
    index code, the gate, the plan, the grounded pipeline and the four grades."""
    from conftest import HarnessKit, make_mcp_config, ok_result

    kit = HarnessKit(make_mcp_config(tmp_path / "cache"))
    read_guide = {"tool": "read_document", "arguments": {"path": "guide.md", "start_line": 1, "max_lines": 400}}
    read_other = {"tool": "read_document", "arguments": {"path": "notes.md", "start_line": 1, "max_lines": 400}}
    draft = {
        "working": "", "answerable": True, "answer": "65536",
        "claims": [{"text": "A list holds 65,536 characters.", "passage_id": "S1-P1", "quote": EVIDENCE_QUOTE}],
    }
    report = {"claims": [{"claim_number": 1, "reason": "stated", "verdict": "supported"}], "answerable": "yes"}
    kit.client.script("mid-model", ok_result(json.dumps({"calls": [read_other, read_guide]})), ok_result(json.dumps(draft)))
    kit.client.script("ground-model", ok_result(json.dumps(report)))

    record = run_eval.run_openbook_case(kit.harness, strategy("mcp"), CASES[0], FakeJudge(), corpus)

    assert record["outcome"] == "pass" and record["mode"] == "mcp"
    assert (record["answer_grade"], record["citation_validity"], record["evidence_hit"]) == ("pass", "pass", "pass")
    assert record["citations"][0]["source"] == "guide.md"
    # notes.md is not one of this case's sources, so it was never staged: the gate refused it.
    assert (record["plan_status"], record["tool_calls_refused"], record["documents_read"]) == ("partial", 1, 1)
    assert [row.error_type for row in kit.tool_call_records if row.status == "refused"] == ["unknown_document"]
    assert not run_eval.Path(kit.tool_sessions.last.root).exists()
