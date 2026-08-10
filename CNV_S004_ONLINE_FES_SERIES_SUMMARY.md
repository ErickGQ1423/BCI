# CNV S004 ONLINE — FES Series Summary

Subject: `CNV_PILOT_SUBJ_014`  
Session: `S004_ONLINE`  
Condition: FES enabled, adaptive recentering enabled  
Start condition: clean adaptive state (`No adaptive transform found — starting fresh`)

## Run 1 — FES from zero

Source log: `/home/lab-admin/.codex/attachments/b6f5742d-c98d-4210-b3e3-0ee70f759fe9/pasted-text.txt`

Key startup checks:

- Adaptive transform: not found at start; initialized from training Riemannian mean.
- FES: active during online preparation / reward (`FES_SENS_GO`, `FES_MOTOR_GO`, `FES_STOP` observed).
- Trial 1: affected by BAD_EEG / `rms_too_high`, endpoint unavailable, final `AMBIGUOUS`.
- Adaptive recentering final state: `accepted_updates=4 | seen=6`.

Main results:

| Metric | Value |
|---|---:|
| Trials | 20 |
| Final total accuracy | 45.0% |
| Final decision accuracy | 56.2% |
| Final correct / incorrect / ambiguous | 9 / 7 / 4 |
| MDM original total accuracy | 45.0% |
| MDM original decision accuracy | 50.0% |
| MDM original correct / incorrect / ambiguous | 9 / 9 / 2 |
| MI recall | 70.0% |
| REST recall | 20.0% |

Confusion matrix, final validated decision:

| Actual | Pred MI | Pred REST | Ambiguous |
|---|---:|---:|---:|
| MI | 7 | 3 | 0 |
| REST | 4 | 2 | 4 |

Endpoint validation:

| Metric | Value |
|---|---:|
| endpoint fallbacks | 14 |
| accepted by LDA | 0 |
| accepted by LR | 1 |
| accepted by both | 9 |
| rejected to ambiguous | 2 |
| errors prevented | 2 |
| correct MDM rejected | 0 |
| MDM already ambiguous | 2 |

Shadow early-stop summary:

| Model | n_trials | early stops | mean step | median step | early-stop accuracy | false MI | false REST |
|---|---:|---:|---:|---:|---:|---:|---:|
| MDM | 20 | 14 | 6.07 | 6.00 | 42.9% | 6 | 2 |
| LDA | 20 | 9 | 6.89 | 7.00 | 66.7% | 3 | 0 |
| LDA3 | 20 | 11 | 7.00 | 7.00 | 63.6% | 1 | 3 |
| LR | 20 | 8 | 6.88 | 7.00 | 75.0% | 1 | 1 |
| SVM | 20 | 4 | 8.00 | 8.00 | 50.0% | 0 | 2 |

Full-window observer summary:

| Model | N | AUC | Accuracy |
|---|---:|---:|---:|
| MDM | 19 | 0.689 | 47.4% |
| LDA_shrink | 19 | 0.633 | 63.2% |
| LDA_shrink_3ch | 19 | 0.667 | 63.2% |
| LR | 19 | 0.589 | 57.9% |
| SVM | 19 | 0.544 | 47.4% |

Interpretation:

- This is the correct first FES run for the new comparison because it starts from the pure offline model/adaptive state, not from the previous no-FES adapted state.
- Compared with mature no-FES runs, performance is lower, but this is expected for run 1 because adaptation restarted from zero.
- The main weakness remains REST: MI recall is acceptable at 70%, but REST recall is only 20%.
- Endpoint validation was helpful in this run: it prevented 2 errors and rejected 0 correct MDM decisions.
- Continue FES runs 2–5 without deleting `adaptive_T.pkl`; the next run should load the FES-adapted transform.

## Run 2 — FES with adaptive transform from Run 1

Source log: `/home/lab-admin/.codex/attachments/7e0ce15e-796b-44d0-8722-b1ef5cca89f2/pasted-text.txt`

Key startup checks:

- Adaptive transform: loaded successfully with `counter = 4`.
- Whitening: loaded from saved adaptive transform.
- FES: active during online preparation / reward.
- Trial 1: again affected by BAD_EEG / `rms_too_high`, endpoint unavailable, final `AMBIGUOUS`.
- Adaptive recentering final state: `accepted_updates=10 | seen=8`.

Main results:

| Metric | Value |
|---|---:|
| Trials | 20 |
| Final total accuracy | 55.0% |
| Final decision accuracy | 61.1% |
| Final correct / incorrect / ambiguous | 11 / 7 / 2 |
| MDM original total accuracy | 55.0% |
| MDM original decision accuracy | 61.1% |
| MDM original correct / incorrect / ambiguous | 11 / 7 / 2 |
| MI recall | 70.0% |
| REST recall | 40.0% |

Confusion matrix, final validated decision:

| Actual | Pred MI | Pred REST | Ambiguous |
|---|---:|---:|---:|
| MI | 7 | 2 | 1 |
| REST | 5 | 4 | 1 |

Endpoint validation:

| Metric | Value |
|---|---:|
| endpoint fallbacks | 12 |
| accepted by LDA | 0 |
| accepted by LR | 3 |
| accepted by both | 7 |
| rejected to ambiguous | 0 |
| errors prevented | 0 |
| correct MDM rejected | 0 |
| MDM already ambiguous | 2 |

Shadow early-stop summary:

| Model | n_trials | early stops | mean step | median step | early-stop accuracy | false MI | false REST |
|---|---:|---:|---:|---:|---:|---:|---:|
| MDM | 20 | 12 | 6.25 | 6.00 | 66.7% | 3 | 1 |
| LDA | 20 | 11 | 6.64 | 6.00 | 27.3% | 5 | 3 |
| LDA3 | 20 | 7 | 7.00 | 7.00 | 42.9% | 2 | 2 |
| LR | 20 | 8 | 6.50 | 6.00 | 25.0% | 5 | 1 |
| SVM | 20 | 4 | 7.00 | 7.00 | 50.0% | 0 | 2 |

Full-window observer summary:

| Model | N | AUC | Accuracy |
|---|---:|---:|---:|
| MDM | 19 | 0.500 | 42.1% |
| LDA_shrink | 19 | 0.311 | 31.6% |
| LDA_shrink_3ch | 19 | 0.378 | 42.1% |
| LR | 19 | 0.344 | 52.6% |
| SVM | 19 | 0.333 | 52.6% |

Interpretation:

- This run correctly continued the FES adaptive series, loading the transform from Run 1.
- Final performance improved from Run 1: total accuracy 45% → 55%, decision accuracy 56.2% → 61.1%.
- REST recall improved from 20% → 40%, which is the most encouraging change.
- MI recall stayed stable at 70%.
- MDM became the best shadow early-stop model in this run: 66.7% early-stop accuracy.
- Endpoint validation was neutral here: it did not prevent errors, but also did not reject correct MDM decisions.

## Running comparison so far

| Run | Start counter | End updates | Final Acc | Decision Acc | MI recall | REST recall | Ambiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|
| FES 1 | 0 | 4 | 45.0% | 56.2% | 70.0% | 20.0% | 4 |
| FES 2 | 4 | 10 | 55.0% | 61.1% | 70.0% | 40.0% | 2 |
| FES 3 | 10 | 12 | 40.0% | 57.1% | 50.0% | 30.0% | 6 |
| FES 4 | 12 | 14 | 50.0% | 66.7% | 70.0% | 30.0% | 5 |
| FES 5 | 14 | 19 | 45.0% | 56.2% | 50.0% | 40.0% | 4 |

Current trend:

- The FES adaptive series is moving in the right direction after Run 2.
- The main signal to watch is REST recall; it doubled from Run 1 to Run 2.
- Continue Runs 3–5 without resetting `adaptive_T.pkl`.

## Run 3 — FES with adaptive transform from Run 2

Source log: `/home/lab-admin/.codex/attachments/4bf1d7f6-ef66-4587-9d50-b681a6b37ae2/pasted-text.txt`

Key startup checks:

- Adaptive transform: loaded successfully with `counter = 10`.
- Whitening: loaded from saved adaptive transform.
- FES: active during online preparation / reward.
- Trial 1: again affected by BAD_EEG / `rms_too_high`, endpoint unavailable, final `AMBIGUOUS`.
- Adaptive recentering final state: `accepted_updates=12 | seen=4`.

Main results:

| Metric | Value |
|---|---:|
| Trials | 20 |
| Final total accuracy | 40.0% |
| Final decision accuracy | 57.1% |
| Final correct / incorrect / ambiguous | 8 / 6 / 6 |
| MDM original total accuracy | 50.0% |
| MDM original decision accuracy | 58.8% |
| MDM original correct / incorrect / ambiguous | 10 / 7 / 3 |
| MI recall | 50.0% |
| REST recall | 30.0% |

Confusion matrix, final validated decision:

| Actual | Pred MI | Pred REST | Ambiguous |
|---|---:|---:|---:|
| MI | 5 | 2 | 3 |
| REST | 4 | 3 | 3 |

Endpoint validation:

| Metric | Value |
|---|---:|
| endpoint fallbacks | 15 |
| accepted by LDA | 0 |
| accepted by LR | 2 |
| accepted by both | 7 |
| rejected to ambiguous | 3 |
| errors prevented | 1 |
| correct MDM rejected | 2 |
| MDM already ambiguous | 3 |

Shadow early-stop summary:

| Model | n_trials | early stops | mean step | median step | early-stop accuracy | false MI | false REST |
|---|---:|---:|---:|---:|---:|---:|---:|
| MDM | 20 | 11 | 6.18 | 6.00 | 45.5% | 4 | 2 |
| LDA | 20 | 14 | 6.14 | 6.00 | 42.9% | 7 | 1 |
| LDA3 | 20 | 7 | 6.29 | 6.00 | 42.9% | 1 | 3 |
| LR | 20 | 12 | 6.00 | 6.00 | 41.7% | 6 | 1 |
| SVM | 20 | 3 | 7.00 | 7.00 | 33.3% | 1 | 1 |

Full-window observer summary:

| Model | N | AUC | Accuracy |
|---|---:|---:|---:|
| MDM | 19 | 0.500 | 47.4% |
| LDA_shrink | 19 | 0.556 | 52.6% |
| LDA_shrink_3ch | 19 | 0.556 | 47.4% |
| LR | 19 | 0.444 | 57.9% |
| SVM | 19 | 0.378 | 42.1% |

Interpretation:

- This run correctly continued the FES adaptive series, loading the transform from Run 2.
- Performance dropped relative to Run 2.
- MDM original was better than the final validated decision: 50.0% vs 40.0% total accuracy.
- Endpoint validation hurt this run: it prevented 1 error but rejected 2 correct MDM decisions.
- This suggests that the LDA/LR endpoint validation layer may be too conservative or unstable during FES adaptation.
- Keep collecting Runs 4–5 before changing anything; the key question is whether Run 3 is fatigue/noise or a systematic effect.

## Run 4 — FES with adaptive transform from Run 3

Source log: `/home/lab-admin/.codex/attachments/78a60d13-8006-4a53-834d-922a6d65362b/pasted-text.txt`

Key startup checks:

- Adaptive transform: loaded successfully with `counter = 12`.
- Whitening: loaded from saved adaptive transform.
- FES: active during online preparation / reward.
- Trial 1: again affected by BAD_EEG / `rms_too_high`, endpoint unavailable, final `AMBIGUOUS`.

Main results:

| Metric | Value |
|---|---:|
| Trials | 20 |
| Final total accuracy | 50.0% |
| Final decision accuracy | 66.7% |
| Final correct / incorrect / ambiguous | 10 / 5 / 5 |
| MDM original total accuracy | 55.0% |
| MDM original decision accuracy | 57.9% |
| MDM original correct / incorrect / ambiguous | 11 / 8 / 1 |
| MI recall | 70.0% |
| REST recall | 30.0% |

Confusion matrix, final validated decision:

| Actual | Pred MI | Pred REST | Ambiguous |
|---|---:|---:|---:|
| MI | 7 | 1 | 2 |
| REST | 4 | 3 | 3 |

Endpoint validation:

| Metric | Value |
|---|---:|
| endpoint fallbacks | 15 |
| accepted by LDA | 0 |
| accepted by LR | 0 |
| accepted by both | 10 |
| rejected to ambiguous | 4 |
| errors prevented | 3 |
| correct MDM rejected | 1 |
| MDM already ambiguous | 1 |

Shadow early-stop summary:

| Model | n_trials | early stops | mean step | median step | early-stop accuracy | false MI | false REST |
|---|---:|---:|---:|---:|---:|---:|---:|
| MDM | 20 | 13 | 6.38 | 6.00 | 38.5% | 5 | 3 |
| LDA | 20 | 10 | 6.20 | 6.00 | 70.0% | 1 | 2 |
| LDA3 | 20 | 12 | 6.42 | 6.00 | 33.3% | 2 | 6 |
| LR | 20 | 14 | 6.64 | 6.50 | 64.3% | 3 | 2 |
| SVM | 20 | 4 | 6.75 | 6.50 | 75.0% | 0 | 1 |

Full-window observer summary:

| Model | N | AUC | Accuracy |
|---|---:|---:|---:|
| MDM | 19 | 0.622 | 52.6% |
| LDA_shrink | 19 | 0.644 | 63.2% |
| LDA_shrink_3ch | 19 | 0.433 | 42.1% |
| LR | 19 | 0.567 | 57.9% |
| SVM | 19 | 0.600 | 57.9% |

Interpretation:

- Run 4 recovered relative to Run 3.
- Final decision accuracy was the best so far in the FES series: 66.7%.
- MI recall recovered to 70%, but REST recall stayed low at 30%.
- Endpoint validation was net helpful in this run: 3 errors prevented vs 1 correct MDM decision rejected.
- MDM original total accuracy was still slightly higher than final validated total accuracy because validation increased ambiguous decisions.

## Run 5 — FES with adaptive transform from Run 4

Source log: `/home/lab-admin/.codex/attachments/042306bf-c69f-46ff-9005-cf74b3eb6bfb/pasted-text.txt`

Key startup checks:

- Adaptive transform: loaded successfully with `counter = 14`.
- Whitening: loaded from saved adaptive transform.
- FES: active during online preparation / reward.
- Trial 1: again affected by BAD_EEG / `rms_too_high`, endpoint unavailable, final `AMBIGUOUS`.
- Adaptive recentering final state observed: `accepted_updates=19 | seen=7`.

Main results:

| Metric | Value |
|---|---:|
| Trials | 20 |
| Final total accuracy | 45.0% |
| Final decision accuracy | 56.2% |
| Final correct / incorrect / ambiguous | 9 / 7 / 4 |
| MDM original total accuracy | 50.0% |
| MDM original decision accuracy | 52.6% |
| MDM original correct / incorrect / ambiguous | 10 / 9 / 1 |
| MI recall | 50.0% |
| REST recall | 40.0% |

Confusion matrix, final validated decision:

| Actual | Pred MI | Pred REST | Ambiguous |
|---|---:|---:|---:|
| MI | 5 | 3 | 2 |
| REST | 4 | 4 | 2 |

Endpoint validation:

| Metric | Value |
|---|---:|
| endpoint fallbacks | 13 |
| accepted by LDA | 0 |
| accepted by LR | 2 |
| accepted by both | 7 |
| rejected to ambiguous | 3 |
| errors prevented | 2 |
| correct MDM rejected | 1 |
| MDM already ambiguous | 1 |

Shadow early-stop summary:

| Model | n_trials | early stops | mean step | median step | early-stop accuracy | false MI | false REST |
|---|---:|---:|---:|---:|---:|---:|---:|
| MDM | 20 | 11 | 6.64 | 6.00 | 63.6% | 3 | 1 |
| LDA | 20 | 12 | 6.50 | 6.00 | 33.3% | 6 | 2 |
| LDA3 | 20 | 11 | 6.73 | 7.00 | 63.6% | 1 | 3 |
| LR | 20 | 13 | 6.85 | 6.00 | 46.2% | 3 | 4 |
| SVM | 20 | 1 | 6.00 | 6.00 | 0.0% | 0 | 1 |

Full-window observer summary:

| Model | N | AUC | Accuracy |
|---|---:|---:|---:|
| MDM | 19 | 0.589 | 57.9% |
| LDA_shrink | 19 | 0.300 | 47.4% |
| LDA_shrink_3ch | 19 | 0.422 | 47.4% |
| LR | 19 | 0.478 | 42.1% |
| SVM | 19 | 0.456 | 31.6% |

Interpretation:

- Run 5 did not continue the recovery seen in Run 4.
- REST recall improved to 40%, but MI recall dropped to 50%.
- MDM original total accuracy was again slightly better than final validated total accuracy.
- Endpoint validation was net helpful but modest: 2 errors prevented vs 1 correct MDM rejected.

## Final 5-run FES summary

| Metric | Mean across 5 FES runs |
|---|---:|
| Final total accuracy | 47.0% |
| Final decision accuracy | 59.5% |
| MI recall | 62.0% |
| REST recall | 32.0% |
| Ambiguous trials / run | 4.2 |

Endpoint validation aggregate across FES runs:

| Metric | Total |
|---|---:|
| Errors prevented | 8 |
| Correct MDM decisions rejected | 4 |

Comparison with no-FES baseline from the same S004 online work:

| Condition | Final Acc | Decision Acc | MI recall | REST recall |
|---|---:|---:|---:|---:|
| No-FES, all 5 runs | 49.0% | 57.9% | 68.0% | 30.0% |
| FES, all 5 runs | 47.0% | 59.5% | 62.0% | 32.0% |
| No-FES, last 2 mature runs | 65.0% | 72.8% | 75.0% | 55.0% |
| FES, last 2 runs | 47.5% | 61.5% | 60.0% | 35.0% |

Final interpretation for today:

- FES did not clearly improve online performance in this self-test.
- Across all 5 runs, FES slightly improved REST recall relative to the full no-FES average, but reduced MI recall and total accuracy.
- The mature no-FES state was clearly better than the mature FES state in this session.
- The strongest scientific result is not that FES helped immediately, but that the system now allows controlled comparison of:
  - pure MDM control,
  - adaptive Riemannian recentering,
  - FES vs no-FES,
  - and passive observer models.
- The recurring Trial 1 BAD_EEG effect remains a real protocol/software issue and should be handled as a warm-up trial or excluded from formal metrics.
