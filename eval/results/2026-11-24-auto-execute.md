# Auto mode chooses execution for computation questions, 2026-11-24

Before its closed-book call, auto mode can ask one local tier a typed yes/no question:
can a short standard-library Python program compute the answer? A confident yes sends
the question to execute mode with the two-program cross-check; anything else is the
closed-book call as before (`auto.computation_check`, shipped on). This file measures
the check on its own and then auto mode end to end on the golden set (36 cases) and the
hard set (26 cases).

Free tiers only: the check on local Ollama models, answers and programs on `mid` (Ollama
cloud), judge `mid`. 0 USD; every model call in the telemetry of these runs went to the
`ollama` provider. One run per case throughout.

Raw records are in `data/eval/` (git-ignored): `check_triage_prompt1.json`,
`check_triage_prompt2.json`, `check_local_prompt1.json`, `check_local_prompt2.json`
(the check alone), `golden_auto_check_run1.json` (strategies `auto` and `auto:check`),
`hard_auto_check_run1.json` (`auto:check`) and the telemetry database
`telemetry_auto_check.db`. Every figure here was recomputed from those records, not
copied from a printed scorecard.

## Targets

| Target | Measured | Met |
|---|---|---|
| Golden score not below the auto baseline run the same day | 35/36 with the check, 35/36 without; the same case missed (DM-06) | yes |
| Hard set: at least 22 of 26 verified and right | 23 verified, 23 right | yes |
| Hard set: 0 verified and wrong | 0 | yes |
| At most 1 golden lookup or domain case sent to execution | 0 (two JavaScript code cases were sent, see below) | yes |
| Median added delay on non-computation questions about 1 second | check call 0.78 s median on the 19 golden questions not sent; 1.19 s median difference in total latency against the baseline on the same 19 | yes |

The targets hold with the check on the `local` tier (qwen3 14b). They do not hold with
the `triage` tier (llama3.2:3b), which was measured on the check alone and not taken
further.

## A. The check alone, 62 labelled cases

Each case carries a `computable` label written before any run (`eval/README.md`): 41
yes (15 golden, 26 hard), 21 no. "Sent" means the check said yes at or above the 0.8
confidence threshold. Run with `scripts/check_computation.py`.

| Check tier | Prompt | Sent, computable | Sent, not computable | Computable, not sent | Not sent, not computable | Precision | Recall | Median latency |
|---|---|---|---|---|---|---|---|---|
| `triage` (llama3.2:3b) | 1 (shipped) | 38 | 9 | 3 | 12 | 0.809 | 0.927 | 241 ms |
| `triage` (llama3.2:3b) | 2 | 37 | 10 | 4 | 11 | 0.787 | 0.902 | 247 ms |
| `local` (qwen3 14b) | 1 (shipped) | 40 | 2 | 1 | 19 | 0.952 | 0.976 | 867 ms |
| `local` (qwen3 14b) | 2 | 39 | 1 | 2 | 20 | 0.975 | 0.951 | 857 ms |

Misclassified ids:

| Check tier | Prompt | Sent wrongly | Missed |
|---|---|---|---|
| `triage` | 1 | LK-01, LK-02, LK-03, LK-05, CD-05, DM-02, DM-04, DM-07, TR-02 | CD-01, HX-11, HX-13 |
| `triage` | 2 | LK-02, LK-03, LK-06, CD-05, CD-07, DM-02, DM-03, DM-04, DM-06, DM-07 | CD-01, HX-11, HX-13, HT-06 |
| `local` | 1 | CD-05, CD-07 | HT-05 |
| `local` | 2 | CD-07 | HR-08, HR-09 |

- llama3.2:3b sends lookups and domain questions to execution (7 or 8 of its wrong
  sends, depending on the prompt) and gives them a confidence of 0.9 to 1.0, so the
  threshold does not filter them. It is too weak for this judgment.
- qwen3 14b sent no lookup, domain or trap case with either prompt. Its wrong sends are
  the two JavaScript code cases, labelled no because a Python program cannot run them.
- No verdict was `low_confidence` or `failed` in any of the four runs: every confidence
  the models gave a yes was 0.8 or higher, so the threshold changed nothing here.
- Prompt 2 is the one revision tried. It adds a test ("a programmer who knows no facts
  and cannot look anything up") and "when unsure, say no". It was written from the
  general failure seen with prompt 1 (facts sent to execution), not from single cases.
  It made llama3.2:3b slightly worse and left qwen3 14b at three errors (one wrong send
  fewer, one miss more), so prompt 1 stays as shipped. Both prompt texts are stored in
  the record files.
- Latency includes one model load: 5.5 s for the first qwen3 call of the prompt 1 run
  and 1.7 s for the first llama call. The medians are of resident models.

## B. Golden set end to end (36 cases)

`auto` is auto mode with the check forced off. `auto:check` forces the check on, asked of
`local`, with program A and program B both on `mid`.

| Strategy | Right | Wrong | Errors | p50 | p95 |
|---|---|---|---|---|---|
| auto (baseline, same run) | 35 | 1 (DM-06) | 0 | 2.6 s | 33.7 s |
| auto:check | 35 | 1 (DM-06) | 0 | 8.9 s | 18.7 s |

No case changed outcome between the two. By category both scored lookup 6/6, code 8/8,
reasoning 9/9, domain 7/8, trap 5/5.

What the check did:

| | Count |
|---|---|
| Sent to execute mode | 17: all 9 reasoning cases and all 8 code cases |
| Of those, labelled not computable | 2 (CD-05, CD-07, the JavaScript cases) |
| Labelled computable and not sent | 0 |
| Lookup, domain and trap cases sent | 0 of 19 |
| Execution status of the 17 | 17 verified, 17 graded right; 0 disputed, 0 one-sided, 0 failed |
| Fell back to the closed-book call | 0 |

- Delay on the 19 questions not sent: the check call took 778 ms median, 966 ms at
  most. Their total latency was 1,187 ms higher than the baseline's at the median of
  the paired differences (the pair also differs by the answer call's own variation).
- Cost in time on the 17 questions sent: median 11.3 s against 3.1 s for the same
  cases in the baseline. The baseline already answered all 17 correctly, so on this set
  the check bought a `verified` status, not accuracy.
- CD-05 and CD-07 were `verified` with the right output (`1,10,100,9` and
  `1,NaN,NaN`): both programs reproduced the JavaScript behaviour in Python. That is
  two programs agreeing on recalled semantics, not a computation of them. It is the
  weak kind of `verified` the cross-check note already warns about.
- The baseline's p95 is two slow closed-book calls (RS-06 59.2 s, TR-01 33.7 s). With
  the check on, RS-06 went to execution.
- 8 of the `auto:check` rubric verdicts are self-graded (the judge is `mid`, which also
  wrote the answer).

## C. Hard set end to end (26 cases)

Compared with the recorded cross-check run with both programs on `mid`
(`data/eval/hard_execute_crosscheck_run2_mid_mid.json`, 2026-11-23), where every case
was sent to execution by flag.

| Run | Sent to execution | Verified (right) | Disputed | One-sided (right) | Closed-book (right) | Answered right | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| crosscheck:mid+mid, recorded | 26 (forced) | 24 (24) | 1 (HX-12) | 1 (HR-10) | 0 | 25 | 32.2 s | 47.5 s |
| auto:check | 25 | 23 (23) | 1 (HT-04) | 1 (HR-10) | 1 (1, HT-05) | 25 | 18.8 s | 33.6 s |

- The check said yes for 25 of 26. The miss is HT-05 (a trap about a TLV message),
  which took the closed-book call and was graded right in 35.6 s.
- No verified answer was wrong: 23 of 23 here, 17 of 17 on the golden set.
- HT-04 was disputed: program A printed `None`, program B `0.00`. It was verified on
  `0` in the recorded run. A disputed run returns no answer and is scored as a fail.
- HX-12, disputed in the recorded run, was verified here on 25775, the expected value.
  Both differences from the recorded run are single cases in single runs.
- HR-10 was one-sided again: one program's draft call failed, the other printed 58.85,
  which is right.
- No run fell back: every question sent to execution produced at least one program
  that ran. 83 sandbox runs across B and C, all removed (`sandbox_removed` 83 of 83);
  no program needed a repair.
- The lower latency than the recorded run is not attributed to anything; the two runs
  were a day apart on a cloud tier.

## Caveats

- One run per case and 62 cases. One hard case is about 4 points; the 23 against 24
  verified is within that.
- The labels were written by the same hand that built the check, before any run, from
  one stated rule. The two JavaScript cases show the rule has a grey zone.
- The hard set is all computable, so it measures recall only. Precision rests on the 21
  golden cases labelled no, of which 19 are the clear kind (lookup, domain, trap).
- The fallback path (execution fails outright, the closed-book answer is returned as
  NOT VERIFIED) did not occur in any live run. It is covered by tests with fakes only.
- `low_confidence` and `failed` verdicts did not occur live either; the threshold of
  0.8 is untested against real output.
- The check runs on a local 14b model that has to be resident: about 0.8 s when it is,
  5.5 s once when it is not (`keep_alive` is 10 minutes).
- The golden run and the hard run record different `config_sha256` values. Only
  comment lines of `config/models.yaml` changed between them.
- Three hard trap cases and the golden rubric cases are judged by `mid`, the model
  that writes the programs and the closed-book answers.
