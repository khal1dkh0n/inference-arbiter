# Phase 2 open-book scorecard, 2026-11-22

26 open-book cases (21 answerable, 5 unanswerable), one run each. Set, corpus and grading
rules: `eval/openbook/README.md`. Openbook = `mid` drafter, hybrid retrieval, `ground_jev`
grounding verifier, `frontier` escalation. Baseline = `stuffed:mid`, one mid-tier call with
every source document pasted before the question. Judge `mid`.

Raw records are in `data/eval/openbook_20261122T161153Z.json` (git-ignored). They are not
copied here because the records quote the fetched third-party pages, which are not
committed.

| Strategy | Answer acc | Abstain acc | False abst | Halluc | Cite valid | Evid hit | Escal | Cost USD | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| stuffed:mid | 21/21 | 5/5 | 0 | 0 | n/a | n/a | 0 | 0.0000 | 2956 | 6012 |
| openbook | 20/21 | 5/5 | 1 | 0 | 20/20 | 20/21 | 0 | 0.0039 | 7522 | 32820 |
| closed:mid | 0/21 | 3/5 | 7 | 2 | n/a | n/a | 0 | 0.0000 | 25911 | 84179 |

Head-to-head, openbook against stuffed:mid: won 0, lost 1 (OB-21), both right 25.

Grounding: 20 runs `grounded`, 6 `ungrounded`. The 6 are the abstentions (OB-21 and the
five unanswerable cases). No run escalated to the frontier tier.

## Findings

- The pipeline matches the baseline less one case, at about 2.5 times the p50 latency and
  a small cost. It does not beat it.
- What the pipeline adds is checked citations: every citation of every answered case was
  verbatim in its source (20/20), and 20 of 21 answerable cases cited the evidence passage.
  The baseline produces no citations to check.
- The baseline saturates this set. The largest case is about 68 KB of sources, which fits
  one mid-tier call. The set can show the pipeline matching the baseline, not beating it. A
  set that can separate them needs a corpus too large to paste.
- OB-21 (two documents, one URL path from each) is a repeatable pipeline miss: 6 passages
  retrieved, the drafter said not answerable, the verifier agreed, the run abstained. Same
  outcome in six pipeline runs out of six (five without the Jev verifier, one with).
- An abstention the verifier agrees with does not escalate. That is right for the five
  unanswerable cases and is why OB-21 never reached the frontier tier.
- `closed:mid` shows the questions need the sources: 0/21. Its 7 false abstentions are
  replies such as "I do not have that document"; its 2 hallucinations are invented values
  (OB-U2, OB-U4).
- 41 rubric and abstention verdicts were made by the model that wrote the answer.
- Small sample: one answerable case is 5 points, one unanswerable case is 20 points, each
  case answered once.

## Phase 2 exit test

The build plan's exit test is scorecard data proving the multi-model chain beats, or
matches at significantly lower cost, a single direct mid-tier call. It is not met on
either suite:

| Suite | Multi-model | Single mid-tier call |
|---|---|---|
| Closed-book, 25 cases (`2026-11-21-phase2-baseline.md`) | chain 22/25, 0.0677 USD, p50 7778 ms | tier:mid 24/25, 0 USD, p50 2765 ms |
| Open-book, 26 cases (this file) | openbook 25/26, 0.0039 USD, p50 7522 ms | stuffed:mid 26/26, 0 USD, p50 2956 ms |

Phase 2 stays open until the exit test is met. Open items at the time of this run: OB-21,
an open-book corpus too large to paste, and the closed-book chain's two lost cases (CD-01,
TR-02). Later runs after those changes: `2026-11-22-phase2-free-runs.md`.
