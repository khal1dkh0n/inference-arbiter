# Codetest mode on the code set, 2026-11-24 (partial)

Codetest mode has one tier write a solution and another write `unittest` tests from the
question and the function signature alone; both run in one throwaway sandbox. This file
measures it on the code set (`eval/code_questions.yaml`, version 1, 20 cases) against a
single direct `mid` call. Every grade comes from the fixed reference tests in
`eval/code_reference/`, which no model sees; the generated tests never decide a grade.

Three strategies were planned. Two finished:

- `tier:mid`: one call, the code taken from the last code fence of the answer.
- `codetest:mid+local`: solution and repair on `mid` (kimi-k3.1-pro:cloud), interface and
  tests on `local` (qwen3 14b). This is the shipped `code_test:` block.

`codetest:mid+mid` did not finish: the run was stopped twice and the runner at that time
wrote its records only at the end, so no record exists for it. Its only trace is a log of 12 cases,
all graded pass by the reference tests, which says nothing about its generated tests.
Whether `mid` writes better tests than `local` is not measured.

One run per case. Every model call went to the `ollama` provider and is logged at 0 USD;
calls to the Ollama cloud model have no price configured, so that figure is not the cost
of the run on a paid Ollama plan. Raw records are in `data/eval/` (git-ignored):
`code_run1_tier_mid.json`, `code_run1_codetest_mid_local.json` and the telemetry database
`telemetry_code_run1.db` (which also holds the rows of the stopped run). Every figure
here was recomputed from those records.

## Result

| | `tier:mid` | `codetest:mid+local` |
|---|---|---|
| Final solution passes the reference tests | 20 of 20 | 20 of 20 |
| First solution passes the reference tests | 20 of 20 | 20 of 20 |
| Latency, median / p95 | 19.2 s / 57.1 s | 28.8 s / 106.7 s |
| Model calls | 20 | 74 (20 interface, 20 solution, 20 tests, 14 repair) |
| Wall time for the 20 cases | 521 s | 1,055 s |

Codetest mode added no accuracy on this set and could not have: `mid` wrote a solution
that passes the reference tests for every case on the first attempt, so there was no
wrong solution for generated tests to catch.

## What the generated tests said

Set against the reference tests (`run_eval.model_tests_verdict`). No solution failed the
reference tests, so "caught" and "missed" are zero by construction.

| Verdict | First solution | Final solution |
|---|---|---|
| clean (generated tests passed, reference passed) | 5 | 6 |
| false alarm (generated tests failed, reference passed) | 14 | 13 |
| untested | 1 | 1 |
| caught / missed | 0 / 0 | 0 / 0 |

- The tool reported `tests_failed` for 13 of 20 solutions that pass the reference tests.
- In all 14 first-attempt false alarms the same generated tests also fail the case's
  reference solution: the tests were wrong, not the code.
- The untested case is CQ-09: the tests call on `local` timed out, and the answer said
  "NOT TESTED".
- 14 repairs ran, each against tests that reject the reference solution. None turned a
  reference-passing solution into a failing one. One (CQ-20) produced a solution that
  passes both the generated tests and the reference tests, although those generated tests
  reject the reference solution.

## Seeded wrong solutions

Each case has two seeded wrong solutions. A generated suite is counted only when it
passes the case's reference solution; otherwise its failures mean nothing.

- 5 of 20 suites qualify (CQ-12, CQ-13, CQ-16, CQ-17, CQ-19); 14 reject the reference
  solution and 1 case has no suite.
- Those 5 suites fail 8 of their 10 seeded wrong solutions. The two that passed are one
  each in CQ-13 and CQ-17.

Five suites are too few to state a catch rate.

## Reading

- With the shipped `tests_tier: local`, the status of a codetest run is unreliable on
  this set: a `tests_failed` was a wrong test 13 times out of 13.
- The reference grading is what showed this. Read without it, the same run looks like 13
  broken solutions and one repair that fixed a bug.
- The set cannot show a benefit of the mode, because the baseline makes no mistake on it.
  A set on which a single call fails some cases is needed for that.
- Auto mode does not pick codetest mode, and nothing here supports changing that.

## Limits

- 20 cases, one run each.
- `codetest:mid+mid` is missing, so the tests tier was not compared.
- Why each generated suite is wrong was not classified case by case; the count rests on
  the suites failing the reference solutions.
- The stopped run ended with exit code 144 both times; the cause was not found.
