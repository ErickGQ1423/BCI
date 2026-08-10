# Notes for next session

Date: 2026-06-18

## Current best pilot configuration

```python
RECENTERING = 1
THRESHOLD_MI = 0.45
THRESHOLD_REST = 0.60
```

Use `master_mdm` for control. Keep LDA and warmup models as observers.

## Metric to trust

Primary metric:

```text
RAW Decision Summary (target-independent)
RAW total accuracy
RAW decision accuracy
RAW counts: correct / incorrect / ambiguous
```

Do not use the old official target-aware accuracy as classifier performance. It is useful for safe reward/control behavior, but it hides wrong opposite evidence as ambiguous.

## What we learned today

- Null/ignore sessions produced roughly 40-50% RAW accuracy.
- Good attentive session reached about 70% RAW with 0 RAW errors.
- Several later sessions dropped, likely fatigue/battery/state instability.
- `RECENTERING = 0` looked worse today:
  - one run around 30% RAW
  - one run around 10% RAW
  - `THRESHOLD_MI = 0.45`, `THRESHOLD_REST = 0.45` improved to 40-50% RAW, but converted ambiguos into errors.
- `THRESHOLD_REST = 0.45` is risky: fewer ambiguos, more accepted wrong decisions.

## Main hypothesis for tomorrow

The final M2 votes may be hurting performance.

There are up to 11 temporal votes per trial. Several trials showed useful early/mid evidence, then the final votes drifted or flipped and degraded the final decision.

Test tomorrow with parallel metrics:

```text
PREP_DECISION_RAW_FULL   = all available votes
PREP_DECISION_RAW_EARLY  = early/mid votes only, e.g. first 5-7 M2 steps
```

Start as logging only. Do not change control until the early-vote metric is clearly better across sessions.

## Suggested next code change

Add an observer-only early voting summary:

- keep current `[PREP_DECISION_RAW]`
- add `[PREP_DECISION_RAW_EARLY]`
- add end-of-session summary for early RAW:
  - early RAW total accuracy
  - early RAW decision accuracy
  - early RAW correct / incorrect / ambiguous

Candidate windows to compare:

```text
steps 1-5
steps 1-6
steps 2-6
steps 2-7
```

Start with one simple choice, probably steps 1-6 or 2-7.
