# Golden Q&A set

Fixed evaluation set for the Phase 2 offline harness (see `inference_arbiter_architecture_build_plan.md`, section 4, Phase 2). Replay it after every prompt, threshold or model change, and score the multi-model chain against a single direct mid-tier call (accuracy, cost, p95 latency).

File: `eval/golden_questions.yaml` (version 2, 36 cases). Version 1 had 25; version 2 (2026-11-22) added 11 harder reasoning, code, domain and trap cases to reach the 30 to 50 the build plan asks for. The added cases were written and their answers computed before any model was run on them; none was chosen because a particular tier failed it. Scorecards from version 1 are not comparable with version 2.

This file covers the closed-book suite. The open-book suite (`scripts/run_eval.py --suite openbook`: questions answered from supplied source documents, with abstention and citation grading) is described in `eval/openbook/README.md`; its `mcp` strategy runs the same questions through mcp mode. Recorded scorecards for both are in `eval/results/`.

## Case schema

Top level: `version` (int) and `cases` (list). Each case:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Unique, stable. Prefix by category: `LK`, `CD`, `RS`, `DM`, `TR`. Never reuse or renumber an id; retire a case by deleting it and logging the removal. |
| `question` | string | Sent verbatim as the user prompt. Self-contained (no attachments, no web access needed). |
| `expected_answer` | string | Short reference answer. Human-readable; the machine check is in `match`. |
| `match` | mapping | Scoring rule, see below. Always has `type`. |
| `category` | string | `lookup`, `code`, `reasoning`, `domain`, `trap`. |
| `expected_route` | string | `local`, `code`, `mid` or `frontier`. The route the owner expects the triage to pick. A hypothesis for measuring routing quality, not part of correctness. |
| `difficulty` | string | `easy`, `medium`, `hard`. |
| `notes` | string | Why the case is in the set and how it was verified. |
| `computable` | bool, optional | Whether a short standard-library Python program can compute the answer from what the question gives. The label the computation check is scored against (see "Computation check labels"). Not read by the loader or the graders. |

## Match types and scoring

Implemented in `src/inference_arbiter/grading.py` (`grade_answer`). A grade is `pass`, `fail` or `ungraded`, always with a reason.

Normalization applied to both the model answer and the reference unless stated: trim, lowercase, collapse runs of whitespace, strip surrounding quotes, backticks, markdown emphasis (`*`, `_`) and trailing punctuation. For `exact`, the grader tries these forms of the answer and passes on the first that equals the reference: the whole answer, the final line (with a leading `Answer:` / `Final answer:` / `Output:` / `Result:` label removed), the content of the last code fence, and the emphasised span (bold or inline code) when the answer has exactly one. Not every `exact` prompt asks for the bare answer (LK-02, CD-01 and CD-05 do not), so explained answers are common; see Adjudication below.

| `match.type` | Fields | Pass rule |
| --- | --- | --- |
| `exact` | none (compares against `expected_answer`) | Normalized answer equals normalized `expected_answer`. For code-output cases, also ignore a surrounding code fence. |
| `contains` | `all_of` (list, each must appear), `any_of` (list, at least one must appear); either or both may be present | Case-insensitive substring match after normalization. Passes only if every `all_of` item is present and, when `any_of` is present, at least one of its items is present. |
| `numeric` | `value`, `tolerance` (absolute) | Parse the final number in the answer (strip thousands separators and currency symbols); pass if `abs(parsed - value) <= tolerance`. A `tolerance` of 0 means exact integer equality. If the answer contains several numbers, use the last one stated as the final result. |
| `reference_tests` | `function`, `tests` (path of a `unittest` file) | Code set only (see "Code set" below). The code in the answer is run in the sandbox against the fixed reference tests; it passes only when every one of them passes. No judge. `grade_answer` itself does not grade this type: without the sandbox path the case is `ungraded`. |
| `rubric` | `criteria` (list of statements), `pass_rule` | Judged by an LLM grader (use a fixed grader model and prompt, temperature 0) or by hand. Each criterion is scored met or not met independently; the case passes only when `pass_rule` holds (currently always: all criteria met). Grader sees the question, the reference answer and the criteria, never the routing trace. |

Details the table leaves open, as implemented:

- `numeric`: a LaTeX `\boxed{}` result wins when present, otherwise the last number in the answer. Currency symbols, thousands separators (`38,111,503`, `38{,}111{,}503`) and times of day (`13:30`) are ignored; a hyphen is a minus sign only when it does not follow a word character, so `1-1000` is not negative. The comparison uses decimal arithmetic, so a tolerance boundary such as 2817.07 against 2817.06 +/- 0.01 is exact. An answer with no number fails.
- `rubric`: the judge tier is a parameter (`--judge-tier`, default `mid`), called at temperature 0 with a JSON schema. The verdict must score every criterion exactly once and its `passed` must equal "every criterion met". A failed judge call, invalid JSON, a missing or duplicated criterion or an inconsistent `passed` is `ungraded`, never pass or fail.
- An empty answer fails. An unknown `match.type` is `ungraded`.

### Adjudication

Position alone cannot grade an explained answer. A model that states the right result and then adds "with a comparator it would print 1,9,10,100" fails the last-number and last-fence rules while being correct. So when an `exact` or `numeric` answer fails the deterministic rule but the expected answer still appears in it (as a number within tolerance for `numeric`; as a whole line, code fence or emphasised span for `exact`), the judge decides one question: is the answer the candidate gives to the question as asked the expected one. The grade keeps its match type and carries `adjudicated: true` plus the deterministic reason. Without a judge such an answer is `ungraded`. A deterministic pass, and a deterministic fail where the expected answer appears nowhere, never reach the judge.

In the first live run (2026-11-21, 80 graded answers) 5 answers were adjudicated; all 5 were correct answers the positional rule had failed, and the judge passed all 5.

Rubric cases carry the most grader noise. Before trusting a Phase 2 scorecard, hand-label the rubric cases once for the baseline model and measure grader agreement. When the judge tier is also the tier being measured (the default `mid` judge on `tier:mid` answers) the model grades itself; the scorecard counts these as self-graded.

## Running the eval

`scripts/run_eval.py` runs strategies over this set, grades them and prints a scorecard. Results go to a timestamped JSON under `data/eval/` and telemetry to `data/eval/telemetry.db`, separate from normal use.

| Strategy | Meaning |
| --- | --- |
| `tier:<name>` | Force one answer tier, route mode. |
| `router` | Triage plus route, route mode. `--triage-tier` selects the triage tier. |
| `chain` | Chain mode. `--verifier-tier` is passed through on the request. |
| `execute:<tier>` | Execute mode, one program written by the tier. |
| `crosscheck:<tier A>+<tier B>` | Execute mode with two independently written programs. |
| `codetest:<solution tier>+<tests tier>` | Codetest mode: one tier writes a solution, the other writes unit tests without seeing it, both run in one sandbox. `codetest:<tier>` takes the tests tier from `code_test.tests_tier`. Meant for the code set. |
| `auto` | Auto mode with the computation check forced off: one call to `auto.closed_book_tier`, with its `on_error` hop. |
| `auto:check` | Auto mode with the computation check forced on: a question the check says a program can compute goes to execute mode, the rest take the closed-book call. The scorecard adds a table of what the check did. |

Options: `--strategies`, `--cases`, `--category`, `--judge-tier`, `--triage-tier`, `--verifier-tier`, `--max-cost-usd` (stops cleanly once cumulative reported cost, answers plus judge, passes it), `--out`, `--db`, `--merge` (include earlier result files in the scorecard and measured routes; this run wins on overlap; a file graded against a different golden set is skipped), `--regrade` (grade the stored answers of a result file again with the current grader, no model answer is re-run), `--routes-out`.

Outcomes per case and strategy: `pass`, `fail`, `ungraded`, `error` (the run failed), `fallback_excluded`. A `tier:<name>` run whose answer came from another tier through the router's fallback hop is `fallback_excluded`: it is flagged, not graded and left out of that tier's accuracy and of the measured routes. For `router` and `chain` the fallback is part of the strategy, so those runs count and are only flagged. Since 2026-11-23 a `tier:<name>` run asks the harness not to fall back (`AskRequest.no_fallback`), so a failed call on the measured tier is an `error` with that tier's error type and no second call is made; `fallback_excluded` still appears in older result files and for `stuffed:`, `closed:` and `execute:` strategies.

Scorecard: per strategy the outcome counts, accuracy over graded cases (pass / (pass + fail)) with a 95% Wilson interval, total and per-question cost, p50 and p95 latency (nearest rank, answered runs only) and accuracy by category; then a head-to-head of each strategy against the baseline `tier:mid` (won, lost, both right, both wrong, cost ratio over the cases both ran). The cost ratio is not available while the baseline cost is zero: `mid` runs on an Ollama cloud subscription and has no per-token prices in `config/models.yaml`.

Each case is answered once and answer calls are not temperature 0, so a one or two case difference between strategies is noise. Latency includes model load time when a local model was not resident.

## Measured routes

`expected_route` is a hand-written hypothesis. From the `tier:*` results `run_eval.py` derives, for each case, the cheapest tier that actually answered correctly and writes `data/eval/measured_routes.json`.

Ordering, cheapest first: `local` and `code` (strength 0, both on this machine at zero cost), then `mid` (strength 1), then `frontier` (strength 2). The `code` tier is only considered for cases in the `code` category, where it is tried before `local`; `acceptable` lists every tier of the cheapest correct strength, so a code case both `code` and `local` got right accepts either.

| Status | Meaning |
| --- | --- |
| `measured` | Cheapest correct tier found and every cheaper tier measured wrong. |
| `provisional` | A correct tier found, but a cheaper tier has no pass/fail result (not run, error, fell back, ungraded). |
| `none_correct` | Every tier was measured and none answered correctly. |
| `unresolved` | No tier answered correctly so far and at least one tier has no result. |

`scripts/compare_triage.py --routes data/eval/measured_routes.json` scores triage against the measured routes as well as `expected_route` and reports both. Cases that are `none_correct` or `unresolved` are left unscored. A measured route rests on one answer per tier, so a tier that is right some of the time can flip a route between runs.

## Computation check labels

Every case of `eval/golden_questions.yaml` and `eval/hard_questions.yaml` carries `computable: true` or `false`. The labels were written on 2026-11-24, before the computation check was run on any case, by one rule: `true` when the answer is a value that follows from data and rules stated in the question, so that a short Python program using only the standard library can compute it; `false` when the answer has to be recalled or judged.

| Set | `true` | `false` |
| --- | --- | --- |
| Golden (36) | 15: every reasoning case (RS-01 to RS-09) and the Python and SQL code cases (CD-01 to CD-04, CD-06, CD-08; `sqlite3` is in the standard library) | 21: every lookup, domain and trap case, and the two JavaScript code cases (CD-05, CD-07: a Python program cannot run them, it can only restate recalled JavaScript semantics) |
| Hard (26) | 26: every case has a reference program under `eval/hard_reference/` | 0 |

`scripts/check_computation.py` asks the check tier about every labelled case and reports the confusion counts, precision, recall, the misclassified ids and the median latency of the check call. No answer is written and nothing is executed. "Positive" means the check said yes at or above `auto.computation_check_threshold`, which is what sends a question to execute mode. `--tier` asks another tier. The result file keeps one record per case and the prompt text used.

Adding the label changed both files' sha256, so `--merge` and `--regrade` refuse result files written before 2026-11-24 for these two sets ("graded against a different golden set"). No question, answer or scoring rule changed, which is why `version` was not bumped.

## Category breakdown

| Category | Count | Ids | Purpose | Usual expected route |
| --- | --- | --- | --- | --- |
| lookup | 6 | LK-01 to LK-06 | Simple facts and definitions a small local model should get right. Guards against needless escalation. | local |
| code | 8 | CD-01 to CD-08 | Single verifiable output of a short Python, SQL or JavaScript snippet. Every answer computed by running it. | code |
| reasoning | 9 | RS-01 to RS-09 | Arithmetic and multi-step problems where small models are often confidently wrong. Every answer computed by code. | mid (RS-05, RS-06 frontier) |
| domain | 8 | DM-01 to DM-08 | Payments (EMV, card flows, ISO 8583, ISO 20022) and agentic AI (MCP, model routing). | mid |
| trap | 5 | TR-01 to TR-05 | False premise, ambiguity, missing context. Correct behaviour is to flag the problem, not to answer. | mid or local |

The brief asked for roughly 7 lookups and 5 domain cases; this set has 6 and 6 so that ISO 20022 and card-flow interchange are both covered.

By route: local 9 (6 lookup, 3 trap), code 8, mid 17, frontier 2. By difficulty: easy 7, medium 23, hard 6.

## Rules for adding or changing cases

1. Answer must be stable over time and checkable without web access. No "latest version" or current-events questions.
2. Compute every numeric and code answer by running it. Record the method in `notes`.
3. If an answer is not certain, drop the case.
4. Keep questions original and self-contained.
5. Keep the file free of secrets; do not paste `.env` contents or keys.
6. Changing a case invalidates comparisons with earlier scorecards. Bump `version` at the top of the YAML when any case is edited or removed, and keep the old file if historical runs must be reproduced.

## Validation

The file must parse as YAML, have 36 cases (or the current agreed count), unique ids, and every case must carry all eight required fields. `match` must have `type`, plus `value` and `tolerance` for `numeric`, `criteria` and `pass_rule` for `rubric`, and `all_of` and/or `any_of` for `contains`.

## Hard set (draft)

File: `eval/hard_questions.yaml` (draft v2, 26 cases, written 2026-11-22, non-executable cases replaced 2026-11-23). A second closed-book set in the same schema, for telling strategies apart once the golden set stopped doing so (a single mid-tier call scores 106/108 on it over three runs). It loads through the same loader:

```
uv run python scripts/run_eval.py --golden eval/hard_questions.yaml --strategies tier:mid --out data/eval/hard_mid_draft.json --routes-out data/eval/hard_measured_routes.json
```

Pass `--routes-out` (and `--db` if the telemetry should stay apart) so a run of this file does not touch the golden set's `data/eval/measured_routes.json`.

| Category | Count | Ids | What it holds |
| --- | --- | --- | --- |
| `executable` | 15 | HX-01 to HX-18 (HX-05, HX-06, HX-09 retired while drafting) | The answer is a value a short standard-library Python program computes from data given in the question: BER-TLV and DOL parsing, an ISO 8583 message, Luhn, half-even fee rounding over a multi-currency batch, CRC-16, XOR LRC, RSA recovery, HMAC-SHA256, SHA-1, zlib, and three debugging cases (HX-10, HX-11, HX-13). |
| `reasoning` | 6 | HR-08 to HR-13 | Problems that need a search or a full enumeration, with the rules stated in the question: final balances over all interleavings, linearizability of five histories, cheapest acquirer routing, concurrent event pairs, the most value a payment queue can settle, serial orders equivalent to a schedule. |
| `domain` | 2 | HD-03, HD-04 | Building an EMV TVR from transaction facts and then running terminal action analysis; a dispute deadline chain over calendar days, business days and holidays. Rules are stated in the question. |
| `trap` | 3 | HT-04 to HT-06 | False premises that have to be checked, not trusted: a batch said to reconcile, a TLV message said to hold an amount, a Java method said to be proven. |

How executable cases are marked: `category: executable`, a category value the golden set does not use. The loader does not validate categories, so `--category executable` selects them and the scorecard reports them as their own column. Because the category is not `code`, the `code` tier is not considered for them in the measured routes.

One field beyond the eight in the schema above, ignored by the loader: `reference`, the path of the program under `eval/hard_reference/` that prints the expected answer. Every executable case has one, and so does every other case whose answer is computed (all eleven reasoning, domain and trap cases). Each reference lists in `QUESTION_DATA` the literals it computes from. `notes` says how each case was verified and what the tempting wrong answers are.

`tests/test_hard_questions.py` loads the file through `run_eval.load_golden`, checks the schema, re-runs every reference program and compares its output with `expected_answer`, checks that each reference's `QUESTION_DATA` literals occur in the question, and checks that the grader passes the expected answer in three phrasings and fails a wrong one.

Grading choices: every computed answer is a decimal number graded `numeric` with tolerance 0, including values that are naturally hex (the CRC, the LRC, the first four bytes of a hash), because a short hex string cannot be graded safely by substring. The one long hex answer (HX-03, 49 bytes) is graded `contains`, with and without spaces between bytes. HR-09, HD-03 and HD-04 ask for a fixed final line (five Y/N letters, a TVR with two outcomes, an ISO date) and are graded `contains` with a few spacings. The three trap cases are `rubric`.

Recalled facts: HT-06 is the only case whose answer rests on a fact recalled and not checked here (Java integer division truncates toward zero). HX-03 and HD-03 state their rules in the question; the rules follow EMV Book 3 as recalled, but the answers depend only on what the question states. The rate tables, the dispute rules, the file format in HT-04 and the certificate layout in HX-14 are invented for the questions.

Retired ids, not reused: HX-05, HX-06, HX-09, HR-01 to HR-07, HD-01, HD-02, HT-01 to HT-03.

### Calibration against the mid tier (2026-11-22, one answer per case)

The first draft (21 cases) scored 20 of 21 on `tier:mid`: `kimi-k3.1-pro:cloud` worked a CRC-16, a float truncation sum and a 49-byte DOL build by hand and got them right. Four cases it answered in seconds were replaced with heavier ones (a 24-row multi-currency batch, a trace of a buggy parser, RSA recovery, twenty Luhn checks) and two reasoning cases were added; the mid tier then passed everything it finished (21 of 23, the other two timed out). Three cases that cannot be done without running code were added last (HX-16 to HX-18).

Final run of the 26-case file (`data/eval/hard_mid_draft.json`): 20 pass, 0 fail, 6 fell back to `local` and were excluded. Five of the six were timeouts (HX-10, HX-14, HX-16, HX-17, HX-18) and one was a provider error (HX-04, `http_5xx`). So the mid tier did not produce an answer of its own for 6 of 26 cases (23%), all executable, and was graded wrong on none. Across the earlier runs it was graded wrong twice: HX-04 once (a parse misaligned by one character; it passed on the next run at 113 seconds) and HX-16 once (it said it had no code execution and returned the program without a value). HX-10 passed in the first run in 67 seconds and timed out in the next two.

What this shows: with all the data in the question, the mid tier's limit on this set is its 120 second timeout, not accuracy. A timed-out `tier:mid` call falls back to `local` and the scorecard reports it as `fallback_excluded`, so it lowers the number of graded cases, not the accuracy figure. Read the "fell back" column together with the accuracy. Results are in `data/eval/hard_mid_draft*.json`.

### Second calibration: harder reasoning, domain and trap cases (2026-11-23)

The first eleven reasoning, domain and trap cases all passed on the mid tier, in 2 to 30 seconds each, so they were retired and replaced with HR-08 to HR-13, HD-03, HD-04 and HT-04 to HT-06. Run of the eleven new cases (`data/eval/hard_mid_v2_nonexec.json`): 9 pass, 0 fail, 2 timeouts (HR-08 and HR-10, the two with the largest search). The executable cases were not run again; their figures above stand.

HR-08 and HR-10 were run once more (`data/eval/hard_mid_v2_rerun2.json`) and timed out again, so on the mid tier they behave like the execution-only cases.

Taken with the run above, the mid tier gave no answer of its own on about 8 of the 26 cases (HX-04's provider error included) and was graded wrong on none. Across some forty different cases tried on it over the two days, it was graded wrong twice. On this model the set separates strategies by time and volume of computation, not by reasoning errors; that is the result to plan around, and a strategy that runs code should recover the timed-out cases.

## Code set

File: `eval/code_questions.yaml` (version 1, 20 cases, written 2026-11-24). Every case asks for one Python function with a stated signature and states the rules the function must follow, including the errors it must raise. It measures codetest mode against a single direct call, and it measures what the tests that codetest mode generates are worth.

```
uv run python scripts/run_eval.py --golden eval/code_questions.yaml --strategies tier:mid codetest:mid+local codetest:mid+mid --out data/eval/code_run1.json --db data/eval/telemetry_code.db --routes-out data/eval/code_measured_routes.json
```

Pass `--routes-out` and `--db` so a run of this file leaves the golden set's routes and telemetry alone. The run needs the `sbx` CLI: grading itself runs code in the sandbox.

First run, 2026-11-24, `tier:mid` and `codetest:mid+local` only: `eval/results/2026-11-24-code-tests.md`. The three strategies together are about 170 model calls and take about 50 minutes; the runner writes its records only when a run ends, so run one strategy per command.

### Case schema

The eight fields of the golden schema, with these differences:

| Field | Meaning |
| --- | --- |
| `id` | `CQ-01` to `CQ-20`. Never reused. |
| `question` | States the signature as one `def` line and every rule the tests rely on. |
| `expected_answer` | A sentence naming the reference tests; not compared with anything. |
| `match` | `type: reference_tests`, `function` (the function's name) and `tests` (the reference test file). |
| `category` | `encoding`, `parsing`, `numeric`, `sequence` or `rules`. |
| `reference_solution` | A right solution, `eval/code_reference/<id>.py`. |
| `broken_solutions` | Seeded wrong solutions, `eval/code_reference/<id>_broken_<n>.py`: the reference solution with one small bug each, named in the file's docstring. Two per case. |

| Category | Count | Ids |
| --- | --- | --- |
| `encoding` | 3 | CQ-01 (Luhn check digit), CQ-04 (CRC-16), CQ-15 (varint) |
| `parsing` | 5 | CQ-02 (BER-TLV), CQ-08 (ISO 8583 bitmap), CQ-09 (strict Roman numerals), CQ-12 (path normalisation), CQ-16 (durations) |
| `numeric` | 3 | CQ-03 (half-even minor units), CQ-11 (RPN with truncating division), CQ-13 (largest-remainder allocation) |
| `sequence` | 5 | CQ-05 (interval merge), CQ-10 (top words), CQ-18 (text wrap), CQ-19 (spiral order), CQ-20 (missing ranges) |
| `rules` | 4 | CQ-06 (IBAN mod 97), CQ-07 (business days), CQ-14 (PAN masking), CQ-17 (version precedence) |

No case of the golden or the hard set is reused: their code cases ask what a snippet prints or for a value, not for a function. Some topics (Luhn, BER-TLV, CRC-16, half-even rounding) are shared with the hard set; the questions and all reference files are new.

### Grading: the reference tests only

Whatever the strategy, pass or fail comes from one thing: the code in the answer, run in the sandbox against the case's reference tests, through the same driver codetest mode uses (`src/inference_arbiter/codetest_driver.py`) and the same gate. For a codetest run the code is the returned solution; for any other strategy it is the content of the last code fence of the answer, and an answer without one fails.

| Reference run | Grade |
| --- | --- |
| Every reference test passed | `pass` |
| A test failed, the code did not load or lacks the function, or the run did not finish | `fail` |
| The sandbox or the gate failed | `ungraded` |

The tests a codetest run generated for itself never decide a grade. No judge model is called for this set.

How the reference files are kept out of every prompt: the request sent to the harness is built from the case's `question` alone; the reference files are read in the grading step, after the answer has come back, and go to the sandbox and nowhere else. `tests/test_run_eval.py` plants a marker in all three kinds of reference file and asserts that it reaches the sandbox and no model call, for a codetest run and for a plain tier run, and that the check fails when a request is built with the reference tests in it.

### What the generated tests were worth

For a codetest strategy each record also says what the run's own generated tests said about the solution, set against the reference tests (`run_eval.model_tests_verdict`):

| Generated tests | Reference tests | Verdict |
| --- | --- | --- |
| failed | failed | `caught` |
| passed | failed | `missed` |
| failed | passed | `false_alarm` |
| passed | passed | `clean` |
| untested | passed or failed | `untested` |

The scorecard prints these counts twice, for the returned solution and for the first solution before any repair, with the catch rate (caught / (caught + missed)) and the false alarm rate (false_alarm / (false_alarm + clean)), each with a 95% Wilson interval. It also prints how many repairs turned a reference fail into a pass (`repair fixed`) and the reverse (`repair broke`): a repair is driven by generated tests, and wrong tests can bend a right solution.

A strong solution tier writes few wrong solutions, so the catch rate above can rest on very few cases. The seeded figures do not depend on the solution tier: each case's generated tests are also run against that case's seeded wrong solutions (caught when they fail one, missed when they pass it) and against its reference solution (tests that fail it are wrong tests).

Every figure is computed from the records in the result file, which keep the solution, the first solution, the generated tests, and every reference and seeded run with its sandbox name and whether the sandbox was removed. These grading runs are not in the telemetry database; the harness's own runs are.

### What the tests prove

`tests/test_code_questions.py` loads the file through `run_eval.load_golden`, checks the schema, and runs every reference file: each case's reference tests pass its reference solution (at least five tests, none skipped, and they do not pass against a stub that raises), and they fail every seeded wrong solution. It also checks that the question states exactly the signature the reference solution has, that every file under `eval/code_reference/` belongs to a case, and that no question quotes an assertion of its reference tests.

### Rules for adding or changing a case

1. State the signature and every rule a test relies on in the question. A behaviour the question leaves open must not be tested.
2. Write the reference solution and the reference tests before any model is run on the case, and do not edit them to make a model's answer pass.
3. Give each case at least two seeded wrong solutions that the reference tests fail, each one small change to the reference solution.
4. Standard library only, no input, no files, no network, a few seconds at most.
5. Bump `version` when a case is edited or removed.
