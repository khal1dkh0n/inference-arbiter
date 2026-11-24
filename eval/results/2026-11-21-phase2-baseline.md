# Phase 2 baseline scorecard, 2026-11-21

25 golden questions, one run each. Chain = local draft, mid verifier, frontier repair
(repair prompt carried the draft, its working and the critique). Raw records:
`2026-11-21-phase2-baseline.json`.

| Strategy | Pass | Fail | Accuracy | Cost USD | p50 ms |
|---|---|---|---|---|---|
| tier:local | 18 | 7 | 72% | 0.0000 | 3961.3 |
| tier:mid | 24 | 1 | 96% | 0.0000 | 2820.9 |
| router | 18 | 7 | 72% | 0.0000 | 3069.9 |
| tier:code | 2 | 3 | 40% | 0.0000 | 12671.7 |
| chain | 20 | 5 | 80% | 0.1890 | 8259.9 |
| tier:frontier | 24 | 1 | 96% | 0.2424 | 3734.0 |

Phase 2 exit test not met: the chain (20/25, $0.19) lost to a single mid-tier call (24/25, $0).
Each repair cost about 2.5x a direct frontier call. All five chain failures were drafts the verifier approved.

## After clean escalation, same day

Corrected golden set (LK-02 states its format, LK-04 accepts equivalent phrasings). Chain = local
draft, mid verifier (Jev for mid drafts), and a rejected draft escalates the original question
alone to the frontier tier. Raw records: `2026-11-21-phase2-clean-escalation.json`.

| Strategy | Pass | Fail | Cost USD | p50 ms |
|---|---|---|---|---|
| tier:mid | 24 | 1 | 0.0000 | 2765 |
| chain | 22 | 3 | 0.0677 | 7778 |

Six escalations at about 0.011 USD each, against about 0.024 with the draft-and-critique repair.
The exit test is still not met: a single mid-tier call is more accurate, faster and free.
