# Phase 2 scorecards after the retrieval, grader and verifier changes, 2026-11-22

Every run in this file is keyless: 0 USD, no OpenRouter call. The open-book pipeline ran
with the local grounding verifier (`--verifier-tier ground_chat`) in place of the default
`ground_jev`, and no frontier escalation was possible. The keyed runs (Jev grounding,
frontier escalation, the closed-book chain) are listed at the end; their results are in
`2026-11-22-phase2-final.md`.

Raw records are under `data/eval/` (git-ignored; they quote fetched third-party pages).
Judge `mid` throughout, so rubric and abstention verdicts on mid answers are self-graded.

Changes since `2026-11-22-phase2-openbook.md`: retrieval fills a 12,000-token budget
instead of a fixed top 6; `stuffed:<tier>` ends in `context_overflow` when the sources do
not fit the tier's context window; the chain verifier also flags ambiguous questions; the
exact-match grader sends an answer that contains the expected text to the judge; the
golden set grew from 25 to 36 cases (version 2).

## Open-book, small set (26 cases, sources fit one call), three runs

Files: `data/eval/openbook_small_run{1,2,3}.json`.

| Strategy | Run | Answer acc | Abstain acc | Cite valid | Evid hit | Escal | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| stuffed:mid | 1 | 21/21 | 5/5 | n/a | n/a | 0 | 2626 | 6498 |
| stuffed:mid | 2 | 21/21 | 5/5 | n/a | n/a | 0 | 2467 | 6821 |
| stuffed:mid | 3 | 21/21 | 5/5 | n/a | n/a | 0 | 2456 | 7306 |
| openbook | 1 | 21/21 | 5/5 | 21/21 | 21/21 | 0 | 23590 | 55468 |
| openbook | 2 | 21/21 | 5/5 | 21/21 | 21/21 | 0 | 21624 | 62803 |
| openbook | 3 | 21/21 | 5/5 | 21/21 | 21/21 | 0 | 24015 | 51299 |

Head-to-head, openbook against stuffed:mid, each run: won 0, lost 0, both right 26.

- OB-21 passes in all three runs. It was a retrieval miss: its two evidence passages ranked
  9 and 25, outside the fixed top 6.
- The pipeline ties the baseline on accuracy. It is about nine times slower here because
  the local verifier reads up to 12,000 tokens per case; the earlier Jev run had a p50 of
  7522 ms at 2,000 to 3,000 tokens.

## Open-book, large corpus (16 cases, sources cannot be pasted), three runs

Corpus: 559 pages, about 14 MB, about 3.5 million estimated tokens, against a mid-tier
context window of 1,048,576 tokens. Files: `data/eval/openbook_large_run{1,2,3}.json`,
`data/eval/openbook_large_closed.json`.

| Strategy | Run | Answer acc | Abstain acc | Cite valid | Evid hit | Escal | Errors | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| stuffed:mid | 1 to 3 | n/a | n/a | n/a | n/a | 0 | 16 of 16 (`context_overflow`) | n/a | n/a |
| openbook | 1 | 13/13 | 3/3 | 13/13 | 13/13 | 0 | 0 | 41004 | 62010 |
| openbook | 2 | 13/13 | 3/3 | 13/13 | 13/13 | 0 | 0 | 41519 | 68899 |
| openbook | 3 | 13/13 | 3/3 | 13/13 | 13/13 | 0 | 0 | 48899 | 69565 |
| closed:mid | 1 | 0/13 | 1/2 | n/a | n/a | 0 | 1 timeout | 42787 | 85906 |

Head-to-head, openbook against stuffed:mid, each run: won 0, lost 0, not comparable 16,
right where the baseline errored 16. These are not counted as wins: the baseline could not
take part.

- The single mid-tier call cannot answer any question over this corpus, with or without
  the sources: pasted, the prompt is over three times its context window; unpasted
  (`closed:mid`), it gets 0 of 13.
- The pipeline answered 39 of 39 answerable runs and abstained on 9 of 9 unanswerable
  runs, every citation verbatim in its source and on the evidence passage.
- Retrieval: every evidence passage ranked within the top 25 of 17,707 passages; a fixed
  top 6 would have lost five of the original cases.
- Leak test. A case is kept only when the mid tier cannot answer it without sources. Six
  drafted cases were dropped for that reason: OB-L01, OB-L10, OB-L12 (answered closed-book
  on the first check) and OB-L05, OB-L07, OB-L09 (each answered once in two graded checks,
  so guessable). Of the 13 kept answerable cases, 10 failed two or three graded closed-book
  runs, OB-L19 failed two, and OB-L14 and OB-L18 failed one each (their earlier
  closed-book attempts timed out and were not graded). One closed-book run is a noisy test;
  a kept case with one or two failures can still be a lucky miss.
- The three dropped guessable cases also passed the pipeline in all three runs of the
  19-case draft (16/16, 3/3 per run), so the score is the same with or without them.
- Small sample: one answerable case is 8 points, one unanswerable case is 33 points.

## Closed-book, golden set version 2 (36 cases), baseline only

Files: `data/eval/golden_v2_mid_run{1,2,3}.json`.

| Strategy | Run | Pass | Fail | Cost USD | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| tier:mid | 1 | 35 | 1 (DM-06) | 0 | 2975 | 32619 |
| tier:mid | 2 | 35 | 1 (DM-06) | 0 | 2678 | 25323 |
| tier:mid | 3 | 36 | 0 | 0 | 3202 | 45978 |

106 of 108. The chain needs the frontier tier for escalation and has not been run on
version 2. It can exceed this baseline by at most two answers in 108, which is inside
run-to-run noise, and it costs more than zero. So on the closed-book suite the best
possible outcome is a tie at higher cost; the keyed chain runs will show whether it
reaches that tie.

## Phase 2 exit test

Exit test: scorecard data proving the multi-model chain beats, or matches at significantly
lower cost, a single direct mid-tier call.

| Suite | Multi-model | Single mid-tier call | Reading |
|---|---|---|---|
| Open-book, large corpus, 16 cases x 3 | 48/48 right | cannot run (0/48); 0/13 without sources | Pipeline does what one call cannot. |
| Open-book, small set, 26 cases x 3 | 78/78 right | 78/78 right | Tie on accuracy; pipeline slower, adds checked citations. |
| Closed-book, 36 cases x 3 | chain not run keyless | 106/108 | At best a tie at higher cost. |

Status at the time of these runs: open. The keyed runs and the verdict are in
`2026-11-22-phase2-final.md`.

## Keyed runs (done on 2026-11-22; results in `2026-11-22-phase2-final.md`)

Each command carries `--max-cost-usd 0.50`. Expected cost is an estimate: Jev grounding
about 0.0008 USD per case at 12,000 tokens, a frontier escalation about 0.01 USD
closed-book and about 0.10 USD open-book.

| Run | Command arguments after `uv run python scripts/run_eval.py` | Expected cost |
|---|---|---|
| Open-book small, Jev | `--suite openbook --strategies openbook --merge data/eval/openbook_small_run1.json --max-cost-usd 0.50 --out data/eval/openbook_small_jev.json` | 0.02 USD, plus 0.10 per escalation |
| Open-book large, Jev | `--suite openbook --golden eval/openbook/questions_large.yaml --corpus eval/openbook/corpus_large.yaml --strategies openbook --merge data/eval/openbook_large_run1.json --max-cost-usd 0.50 --out data/eval/openbook_large_jev.json` | 0.02 USD, plus 0.10 per escalation |
| Closed-book chain, run N of 3 | `--strategies chain --merge data/eval/golden_v2_mid_runN.json --max-cost-usd 0.50 --out data/eval/golden_v2_chain_runN.json` | 0.10 USD per run |
