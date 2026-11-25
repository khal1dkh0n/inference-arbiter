# Open-book eval set

Questions whose answers must come from supplied source documents. The set exists to answer
one question: does the grounded pipeline (retrieve, cited draft, deterministic quote check,
groundedness verifier, one escalation) beat pasting the documents into a single mid-tier
call, on accuracy, citation correctness and refusing to answer when the sources do not
contain the answer?

## Files

- `corpus.yaml` - the source documents and a SHA-256 of each file's bytes.
- `questions.yaml` - the cases (26: small sources, at most about 68 KB per case).
- `corpus_large.yaml`, `questions_large.yaml` - the large-corpus set, see below.
- `../../scripts/fetch_corpus.py` - fetches and verifies the corpus.
- `../../scripts/run_eval.py --suite openbook` - runs and scores.

## Corpus

Two kinds of document:

- Files in this repository (the two spec documents). No `url` in the manifest.
- Third-party documentation pages, chosen because they describe things newer than model
  training data. They have a `url`, are fetched into `data/corpus/` (git-ignored) and are
  never committed. Only the manifest of URLs and hashes is in git.

Commands:

- `uv run python scripts/fetch_corpus.py` fetches any missing remote file and verifies every hash.
- `uv run python scripts/fetch_corpus.py --check` verifies only, with no network.
- `uv run python scripts/fetch_corpus.py --refetch --pin` accepts new upstream content.

The fetch is plain `httpx` with a generic client name and sends nothing else.

The eval refuses to run when a corpus file is missing or its hash differs from the
manifest. Questions and evidence quotes were written against one exact corpus; a changed
page can move or delete the evidence, so results on a different corpus are not comparable.
Each result file records a corpus fingerprint, and `--merge` and `--regrade` reject files
produced against another corpus or question file. After `--pin`, run `uv run pytest`:
`tests/test_openbook_eval.py` checks every evidence quote against the files present, and
`run_eval.py` does the same at start-up.

The build plan was pinned again on 2026-11-25 after a companion file name in it was
brought up to date. No question and no evidence quote changed, but the corpus
fingerprint did, so `--merge` and `--regrade` refuse open-book result files written
before that date.

## Large-corpus set

`corpus_large.yaml` lists pages of the OpenRouter documentation named in its `llms.txt`
index on 2026-11-22. The reported runs used all 559 pages of that index: about 14 MB,
about 3.5 million estimated tokens in 17,707 passages. The manifest now lists 551 of them.
That is more than three times the mid tier's context window (1,048,576
tokens), so the documents cannot be pasted into one call. `questions_large.yaml` holds 16
cases (13 answerable, 3 unanswerable), each asked of the whole corpus with `sources: all`.

    uv run python scripts/fetch_corpus.py --manifest eval/openbook/corpus_large.yaml --pause-s 0.25
    uv run python scripts/run_eval.py --suite openbook --golden eval/openbook/questions_large.yaml --corpus eval/openbook/corpus_large.yaml --strategies openbook stuffed:mid closed:mid

The first pipeline run embeds every passage with the local embedding model and caches the
vectors under `data/cache/embeddings/`; later runs read the cache. `stuffed:mid` ends in
`context_overflow` on every case without calling a model. The scorecard shows those runs as
errors and, in the head-to-head, counts the pipeline's correct answers on them in their own
column ("right, baseline errored"). They are never added to "won": a baseline that cannot
run was not beaten on a question, it was unable to take part.

The unanswerable cases of this set were checked by searching the fetched pages for the
value asked about. Each has a near miss elsewhere in the corpus (another router's time
budget, another engine's image cap), which is what makes abstaining the hard part.

## Case schema

| Field | Meaning |
|---|---|
| `id` | `OB-nn` answerable, `OB-Un` unanswerable. |
| `question` | What is asked. |
| `sources` | Corpus document ids supplied with the question (one or several), or `all` for every document of the manifest. |
| `answerable` | `false` when the supplied sources are on-topic but do not contain the answer. |
| `expected_answer` | Reference answer, or what the sources do and do not say. |
| `match` | `exact`, `contains`, `numeric` or `rubric` as in `eval/README.md`; `abstain` for unanswerable cases. |
| `evidence` | List of `{document, quote}`: the supporting text, copied verbatim from the source. Empty for unanswerable cases. |
| `category` | `lookup`, `exact_value`, `contradicts_prior`, `multi_source`, `unanswerable`. |
| `notes` | Why the case is in the set. |

Rules for a case:

- An answerable case must be answerable only from its sources. Each one was asked of the mid
  tier with no sources (`closed:mid`) and kept only when that answer was wrong.
- Every evidence quote is present verbatim in its document after whitespace normalisation,
  and its document is one of the case's sources.
- Unanswerable cases are realistic questions about the supplied documents, not tricks.

## Dropped cases

- OB-17 ("Is openrouter_metadata present on a response served from cache when I send the
  opt-in header?", source router_metadata, expected: no, cache hits never include it).
  Dropped on 2026-11-21 because `closed:mid` answered it correctly with no sources, so it
  does not test use of the sources. The id is not reused.

- OB-15 ("With router metadata enabled, does a response with HTTP status 500 include
  openrouter_metadata? What about a 503?", source router_metadata, expected: 500 no, 503
  yes). Dropped on 2026-11-21: `closed:mid` failed it on the first run and passed it on the
  second. Two yes/no parts can be guessed, so it does not reliably test use of the sources.

- OB-L01 (how many top-spend models the default Switchyard pair is chosen from: 20),
  OB-L10 (the credit purchase fee: 5.5 percent, 0.80 USD minimum) and OB-L12 (sticky
  session expiry: 60 minutes, and 5 minutes for the Pareto Code Router). Dropped from the
  large-corpus set on 2026-11-22: `closed:mid` answered all three correctly with no
  sources. The ids are not reused.

- OB-L05 (Fusion cost with the default panel: 4 to 5 times, 3 models), OB-L07 (Pareto Code
  Router same-tier fallbacks: up to two) and OB-L09 (mistral-ocr image cap: 8 per PDF).
  Dropped from the large-corpus set on 2026-11-22: over two `closed:mid` runs each was
  answered correctly once with no sources, so each can be guessed. The ids are not reused.
  Of the 13 kept answerable cases, 10 failed two or three graded `closed:mid` runs, OB-L19
  failed two, and OB-L14 and OB-L18 failed one each (their earlier attempts timed out and
  were not graded). No kept case was answered closed-book in any run.

## Strategies

| Strategy | What runs |
|---|---|
| `openbook` | The grounded pipeline: `AskRequest(mode="openbook", sources=[...])`. |
| `stuffed:<tier>` | One call to the forced tier with every source document pasted before the question (`mode="stuffed"`). While the harness has no stuffed mode the runner pastes the sources itself and says so under the scorecard. |
| `closed:<tier>` | One call to the forced tier with no sources: what memory alone gets. |
| `mcp` | Mcp mode: `AskRequest(mode="mcp", docs_root=...)`. The case's documents are copied into a temporary folder under their own file names, served from there by the read-only documents server behind the tool gate, and answered by the same grounded pipeline as `openbook`. Not in the default list; name it with `--strategies`. |

Notes on `mcp`:

- A case is asked of its own documents only, as with the other strategies: the staged
  folder holds exactly the case's sources and is removed after the run.
- One extra model call per case: the tool plan (which documents to read), written by
  `mcp_docs.plan_tier` from the question and the document names.
- Citations name the document by file name, which the grader resolves when the name is
  unambiguous among the case's sources. Citation validity and evidence hit are graded as
  for `openbook`.
- Each record also keeps `plan_status`, `documents_listed`, `documents_read`,
  `tool_calls_total`, `tool_calls_refused`, `tool_calls_from_plan` and
  `tool_server_stopped`.
- Not for the large-corpus set: every chosen document is read in full through tool calls
  on every run.
- No full scorecard yet. A first run of three cases with a live model is recorded in
  `eval/results/2026-11-24-mcp-smoke.md`: two passed, and one abstained on an answerable
  case because the plan left out the document that held the answer.

## Grading

Four separate grades per run. They are reported as separate columns and never combined
into one number. Implementation: `grade_openbook` in `src/inference_arbiter/grading.py`.

1. **Answer** (answerable cases only). The case's `match` rule, exactly as closed-book
   grading. A run that abstained on an answerable case fails.
2. **Abstention**: did the run make the right answer-or-abstain decision?
   - Unanswerable case: pass only when the run abstained.
   - Answerable case: abstaining is a fail; answering is a pass (the answer grade says
     whether the answer was right).
   - How abstaining is detected: the run's own `abstained` flag when it is true. Otherwise a
     correct answer is not an abstention, and any other answer is read by the judge with one
     fixed criterion (the answer plainly says the sources do not contain what is needed and
     does not commit to a specific answer). With no judge, or a failed judge call, the grade
     is ungraded.
3. **Citation validity** (deterministic). Every citation's `quote` must appear in the text
   of the cited source after whitespace normalisation, and the cited source must be one of
   the documents supplied for the case. Case and punctuation are not normalised: a quote is
   verbatim or it is invalid. An answer to an answerable case with no citation fails. The
   grade is not applicable when there is nothing to check: the run abstained, or the
   strategy does not cite (`stuffed`, `closed`). Validity says the quoted text exists; it
   does not say the quote supports the claim.
4. **Evidence hit** (deterministic, answerable cases). Pass when some citation cites the
   evidence document and its quote shares at least four consecutive words with an evidence
   quote (or wholly contains an evidence quote shorter than that), or the passage the
   citation came from contains the evidence quote, when the pipeline reports
   `passage_text`. One hit is enough; the record keeps how many evidence quotes were hit.

A citation's `source` is matched to a document by id, path, URL or unambiguous file name.

## Scorecard

One row per strategy:

| Column | Numerator / denominator |
|---|---|
| answer acc | correct answers / answerable cases with a pass or fail |
| abstain acc | correct abstentions / unanswerable cases with a pass or fail |
| false abst | count of answerable cases the run declined |
| halluc | count of unanswerable cases the run answered |
| cite valid | runs with all citations valid / runs with a citation verdict |
| evid hit | answerable runs with an evidence hit / answerable runs with a verdict |
| esc tried | count of runs where the pipeline called the stronger tier |
| esc ok | of those, runs whose stronger-tier answer was used (`repaired`) |
| esc failed | of those, runs whose stronger-tier call gave nothing usable; the draft was kept (`repair_error_type`) |
| cost usd, p50 ms, p95 ms | as in the closed-book scorecard |

A table under the scorecard gives the cost of the escalation calls and how much of it went
on failed ones (a failed call is still billed). Result files written before 2026-11-22 do
not carry the per-call cost; when merged or regraded it is read from the telemetry
database by trace id, and shown as "unknown" when the run is not found there.

Results: `eval/results/2026-11-22-phase2-final.md` (keyed runs, both sets) and
`eval/results/2026-11-22-phase2-free-runs.md` (three keyless runs per case).

Ungraded, error and fallback-excluded runs are left out of every rate and shown in their
own columns. The head-to-head compares `openbook` with `stuffed:mid` per case, where a case
is right when the answer is correct or, for an unanswerable case, when the run abstained.

Small sample: with about 20 answerable cases one case moves answer accuracy by about 5
points, and with 5 unanswerable cases one case moves abstention accuracy by 20 points. Each
case is answered once. Differences of one or two cases are within noise.

## Known limits

- The judge tier defaults to `mid`, so `stuffed:mid` and `closed:mid` answers to rubric and
  abstention checks are judged by the model that wrote them. The scorecard counts these as
  self-graded. Use `--judge-tier` to change the judge.
- Citation validity is strict about characters. A model that tidies quotation marks or
  drops Markdown backticks in a quote produces an invalid citation.
- Evidence hit depends on the hand-picked evidence quote. A correct answer can rest on
  another passage that states the same fact; that reads as a miss.
