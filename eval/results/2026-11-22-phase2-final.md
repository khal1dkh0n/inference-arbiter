# Phase 2 final scorecard and exit-test verdict, 2026-11-22

Keyed runs with the shipped defaults: `ground_jev` grounding and `frontier` escalation for
the open-book pipeline, the verification chain with `frontier` repair for the closed-book
suite. Judge `mid`. The keyless runs these build on are in `2026-11-22-phase2-free-runs.md`.

Raw records: `data/eval/openbook_small_jev.json`, `openbook_large_jev.json`,
`golden_v2_chain_run{1,2,3}.json` and the baselines `golden_v2_mid_run{1,2,3}.json`,
`openbook_small_run1.json`, `openbook_large_run1.json` (git-ignored; they quote fetched
third-party pages). Escalation counts and costs were recomputed from these records and
the telemetry database (`data/eval/telemetry.db`), not from the scorecard printed at run
time, whose single "escal" column counted only escalations whose answer was used.

Total keyed spend: 0.9605 USD (chain 0.4833, open-book 0.4772).

## Closed-book, golden set version 2 (36 cases), three runs

| Run | chain pass | chain cost USD | Escalations tried / ok / failed | chain p50 ms | chain p95 ms | tier:mid pass | tier:mid p50 ms | Head-to-head (chain) |
|---|---|---|---|---|---|---|---|---|
| 1 | 33/36 | 0.1576 | 14 / 14 / 0 | 11708 (inflated) | 47713 (inflated) | 35/36 | 2975 | won 0, lost 2 (DM-04, TR-02), both wrong 1 (DM-06) |
| 2 | 34/36 | 0.1703 | 14 / 14 / 0 | 9891 | 65059 | 35/36 | 2678 | won 0, lost 1 (TR-02), both wrong 1 (DM-06) |
| 3 | 35/36 | 0.1554 | 13 / 13 / 0 | 10674 | 79772 | 36/36 | 3202 | won 0, lost 1 (DM-06) |
| Total | 102/108 | 0.4833 | 41 / 41 / 0 | | | 106/108, 0 USD | | won 0, lost 4 |

- Run 1 ran at the same time as four other eval runs on the same machine and the same
  providers, so its latency is inflated and not comparable with runs 2 and 3.
- Nearly all of the chain's cost is frontier repair: 0.4824 of 0.4833 USD, about 0.012 USD
  per escalation, on 13 or 14 of 36 cases per run.
- The chain never answered a case the baseline got wrong. Its four losses: DM-04 (run 1,
  the frontier repair introduced an error), TR-02 (run 1 after repair, run 2 a local draft
  the verifier approved), DM-06 (run 3, a mid draft the verifier approved).
- Self-graded verdicts (judge is the model that wrote the answer): chain 4, 3 and 4;
  tier:mid 10, 10 and 11.

## Open-book, small set (26 cases, sources fit one call)

| Strategy | Answer acc | Abstain acc | Cite valid | Evid hit | Escalations tried / ok / failed | Cost USD | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| stuffed:mid (keyless run 1) | 21/21 | 5/5 | n/a | n/a | n/a | 0 | 2626 | 6498 |
| openbook, Jev grounding | 21/21 | 5/5 | 21/21 | 20/21 | 1 / 0 / 1 | 0.0872 | 8241 | 25244 |

Head-to-head: won 0, lost 0, both right 26. Grounding: 20 grounded, 5 ungrounded (the five
abstentions), 1 partly grounded (OB-23). Of the 0.0872 USD, 0.0790 is the one failed
frontier escalation; Jev grounding for all 26 cases cost about 0.008 USD.

## Open-book, large corpus (16 cases, sources cannot be pasted)

| Strategy | Answer acc | Abstain acc | Cite valid | Evid hit | Escalations tried / ok / failed | Errors | Cost USD | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| stuffed:mid | n/a | n/a | n/a | n/a | n/a | 16 of 16 (`context_overflow`) | 0 | n/a | n/a |
| closed:mid (keyless) | 0/13 | 1/2 | n/a | n/a | n/a | 1 timeout | 0 | 42787 | 85906 |
| openbook, Jev grounding | 13/13 | 3/3 | 13/13 | 13/13 | 2 / 1 / 1 | 0 | 0.3900 | 15112 | 26990 |

Head-to-head against stuffed:mid: won 0, lost 0, right where the baseline errored 16.
Grounding: 12 grounded, 3 ungrounded (the three abstentions), 1 partly grounded (OB-L03).
Of the 0.3900 USD, 0.3813 is two frontier escalations (0.1852 used, 0.1961 failed); Jev
grounding for all 16 cases cost about 0.009 USD.

## Exit-test verdict

Exit test (build plan): scorecard data proving the multi-model chain beats, or matches at
significantly lower cost, a single direct mid-tier model call on the 30 to 50 question
suite.

| Suite | Verdict | Why |
|---|---|---|
| Closed-book, 36 cases (the suite the exit test names) | Not met | Chain 102/108 for 0.4833 USD against tier:mid 106/108 for 0 USD. Fewer right answers, higher cost, about four times the latency, no case won. |
| Open-book, small set | Not met | Equal accuracy (26/26 each). The pipeline costs 0.0872 USD against 0 and is about three times slower. It adds checked citations, which the exit test does not measure. |
| Open-book, large corpus | Met only in the sense that the baseline cannot run | Pipeline 16/16; a single mid-tier call cannot take the sources (over three times its context window) and gets 0/13 without them. No question was won head-to-head, because the baseline answered none. |

Phase 2 as written is not passed: on the suite the exit test names, a single mid-tier call
is more accurate and free. The multi-model path is ahead only where one call cannot hold
the sources.

### Decision: Phase 2 is closed on this result

The hypothesis the exit test was written to confirm did not hold. For closed-book
questions, and for sources that fit in one call, a single mid-tier call is the better
choice, and the harness now does exactly that by default (`auto` mode). Phase 2 is closed
on what the measurements did show:

- Multi-model work pays where one call cannot hold the sources: 16/16 on the large corpus
  against 0/13 without them, with every citation checked against its source.
- It pays where the answer needs computation: see `2026-11-23-hard-set.md`, where sandboxed
  execution answers 25 of 26 hard cases and a two-program cross-check verifies 24 with none
  wrong, against 18 and 20 for single mid-tier and frontier calls.
- It does not pay as a general verification chain over a strong mid tier. The chain is
  kept behind a flag and is not the default.

## Caveats

- Failed escalations. Three frontier escalations were attempted in the open-book runs and
  two failed: OB-23 (truncated, invalid JSON) and OB-LU1 (no content). Both were billed
  (0.0790 and 0.1961 USD, 0.2751 of the 0.4772 USD open-book spend) and the pipeline kept
  the mid draft, which was graded correct both times. So no open-book answer in these runs
  depended on a frontier escalation except OB-L03, and the frontier path worked once in
  three attempts.
- Jev's partly grounded verdicts. Jev marked two drafts partly grounded (OB-23, OB-L03)
  and both answers graded correct, OB-23 on the kept mid draft. Each triggered a frontier
  call of 0.08 to 0.19 USD that the grade did not need. Two of two is too few to call a
  false-alarm rate. OB-23 is also the one evidence miss: its citations were valid but none
  was on the evidence quote.
- Self-grading. Judge and answering model are the same `mid` model for 5 open-book
  verdicts on the small set, 3 on the large set, 11 chain verdicts and 31 tier:mid
  verdicts across the three closed-book runs.
- Leak test. Of the 13 answerable large-corpus cases, OB-L14 and OB-L18 have only one
  graded closed-book check each (earlier attempts timed out); OB-L19 has two; the other 10
  have two or three. All were answered wrongly without sources.
- The keyed open-book runs are one run per case. The three-run figures (78/78 and 48/48)
  come from the keyless runs with the local verifier.
- Small samples: one answerable case is 5 points on the small set and 8 on the large one;
  one unanswerable case is 20 and 33 points.
- Chain run 1 latency is inflated by concurrent runs.

## Addendum: escalation check after the abstention fix (same day)

Two large-corpus cases rerun alone with the keyed defaults, after telemetry began recording
each call's finish reason (schema 4) and abstentions began escalating only on a positive
verifier verdict. Raw records: `data/eval/openbook_large_escalation_check.json` (git-ignored).

| Case | Jev verdict | Frontier call | Result | Escalation USD |
|---|---|---|---|---|
| OB-L03 | claim 2 uncertain (p=0.46): partly grounded | ok, finish `stop`, 21,264 prompt tokens | frontier answer used, graded correct | 0.1864 |
| OB-LU1 | "passages answer the question" yes (p=0.77) | refused, finish `content_filter` (native `refusal`), 23,761 prompt tokens | mid draft kept, correct abstention | 0.1952 |

- The failed frontier replies are provider refusals, not truncation by the output cap and
  not load: OB-LU1 was refused again when run alone. The refusal text cites terms on
  reverse engineering or duplication; the input was OpenRouter's public documentation
  passages and the question. OB-23's truncated reply in the earlier run was not recorded in
  enough detail to confirm, and fits the same cause.
- A refused call is billed for its input. Across the keyed open-book runs, 5 frontier
  escalations were attempted, 2 used, 3 failed, and the failed ones cost 0.470 USD.
- OB-LU1 still escalates under the new rule because Jev positively (p=0.77) rated an
  unanswerable question answerable. The sources state a 3-second routing budget for a
  different router. Of 16 Jev answerability checks on abstained drafts across the keyed
  runs, the two on OB-LU1 are the only ones above 0.3 (the earlier value was not stored);
  the 0.7 threshold was not changed for one case.
- Follow-up in progress: a smaller escalation input (cited passages plus the top-ranked
  few, under a token cap) so that a refused or unneeded escalation costs less.

## Addendum 2: capped escalation input (same day)

The same two cases rerun alone after `openbook.escalation_token_budget: 3000` shipped (the
escalation gets the cited passages plus the best-ranked ones under the cap) and refusals
became a final `refused` outcome. Raw records:
`data/eval/openbook_large_escalation_capped.json` (git-ignored).

| Case | Jev verdict | Escalation input | Frontier call | Result | Escalation USD |
|---|---|---|---|---|---|
| OB-L03 | claim 2 uncertain (p=0.60): partly grounded | 13 passages, 2,973 estimated tokens | ok, finish `stop`, 5,893 prompt tokens | frontier answer used, graded correct, citations valid | 0.0636 |
| OB-LU1 | "passages answer the question" uncertain (p=0.66) | none | not called | mid draft's abstention kept, correct | 0 |

- OB-L03's escalation fell from 21,264 to 5,893 prompt tokens and from 0.1864 to 0.0636 USD,
  with the same correct answer. One case, one run.
- OB-LU1 did not escalate this time. Jev's answerability probability for it was 0.77 in the
  previous check and 0.66 here on the same 12,986-token input, so it sits near the 0.7
  threshold and will escalate on some runs. The refusal path therefore has not been
  exercised live since it was typed as `refused`; it is covered by tests only.
- Whole run: 0.0647 USD against 0.3827 USD for the same two cases before the cap.
