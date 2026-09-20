# When One Critical Coupling Is Not Enough: Residual Bypass Desynchronizes Representation, Dynamics, and Utility

**Evidence-bounded manuscript draft — 20 September 2026**

## Abstract

Many analyses of learned bottlenecks summarize learning with a single critical control value, denoted here by `C*`. That summary is only adequate if representation geometry, internal dynamics, and downstream utility change at the same point. We test this assumption in a synthetic symbolic-bottleneck protocol with three operational phase markers: rank/equivalence (`C*_rank`), dynamics (`C*_dyn`), and downstream utility (`C*_util`).

In D4a, an eight-cell formal alignment table covers two splits, two channel variants, two bottleneck sizes, and 20 seeds per cell. The archived integrated report gives mean absolute separations of 0.312 between `C*_dyn` and `C*_util`, 3.212 between a null-dynamics marker and utility, and 12.769 between `C*_rank` and utility; the dynamics marker is closer to utility than the rank/equivalence marker in 93.8% of the reported comparisons. In D4b, the archived grouped summary covers 3,000 configurations, 10 seeds, five residual strengths, and three splits. Introducing a continuous residual bypass is accompanied by a rise in `sync_gap = |C*_dyn - C*_rank|` from 2.60–3.95 at zero residual to 7.50–13.60 in the reported nonzero-strength groups, while utility sensitivity falls from approximately 0.47 to 0.03–0.21. The rate at which dynamics appears before rank/equivalence also increases from 0.25 at zero residual to as high as 0.80–0.85.

The result is not that one universal critical constant has been discovered. In the current implementation, each point on the `C` curve is produced by a fresh training run, and the threshold is extracted from a discrete grid after local smoothing. The defensible conclusion is narrower: within this synthetic protocol, rank/equivalence, dynamics, and utility are separable phase markers; the dynamics marker is the most utility-aligned of the tested structural markers; and residual bypass is associated with a reordering of those markers. A fixed-model/common-random-number sweep, formal raw D4b export, and mechanism controls are required before making a stronger causal claim.

## 1. Introduction

The phrase “the critical coupling” compresses several distinct questions. At what control value does a learned representation become symbolically equivalent? At what value does it support the relevant state transition or dynamical law? At what value does the downstream task become useful? These questions can have different answers even when they use the same scalar control parameter. This framing is related to work showing that learning dynamics can exhibit plateaus and rapid transitions [1], and to analyses that treat representation changes as structured rather than as a single scalar event [3, 4].

This distinction matters whenever a bottleneck is intended to mediate both representation and action. A downstream readout can become effective through a continuous shortcut before the bottleneck has acquired the intended symbolic structure. Residual parameterizations are a natural architectural context for such a path because they explicitly add an input-referenced residual function [2]. Conversely, a representation can look geometrically organized while its transition dynamics remain unusable. Treating all of these events as one `C*` can therefore conceal the order in which a system acquires different capabilities.

We study this problem in a controlled synthetic setting. The experiment has two parts. D4a decomposes the operational threshold into rank/equivalence, dynamics, and utility markers and compares their alignment. D4b changes the strength of a continuous residual path while keeping the nominal symbolic pathway and evaluation protocol fixed at the design level, then asks whether the phase markers separate.

The paper makes three contributions.

1. It replaces a scalar-threshold question with an explicit phase-marker decomposition.
2. It reports an empirical alignment result: the dynamics marker is closer to utility than the rank/equivalence marker in the archived D4a result.
3. It turns the residual bypass into a desynchronization hypothesis and reports the associated ordering statistics across ID and OOD splits.

The paper also makes the limits of these contributions explicit. The current D4b archive preserves the formal grouped summary but not the formal per-configuration CSV. More importantly, the implementation trains a new model separately for every `C`, so the current curves estimate an operational threshold under per-`C` retraining. They do not yet identify the effect of changing `C` in a fixed trained system. These limitations shape the claim ceiling throughout the manuscript.

## 2. Research question and experiment card

### 2.1 Research question

Does one operational `C*` summarize representation geometry, dynamics structure, and downstream utility in a symbolic bottleneck? If not, does a continuous residual bypass change the ordering of these phase markers?

### 2.2 Actively changed variables

- The control grid `C`, used to obtain operational phase markers.
- Residual bypass strength `r ∈ {0, 0.1, 0.25, 0.5, 1.0}` in D4b.
- Split (`id`, `ood_random`, `ood_inverted` in D4b; `id`, `ood_inverted` in D4a).
- Bottleneck size `K ∈ {8, 16}`.
- Channel variant (`original`, `pre_softmax`) in D4a; the formal D4b summary is reported for the original variant.

### 2.3 Target quantities

For each condition, the protocol extracts three grid-derived thresholds:

- `C*_rank`: first grid point at which the rank/equivalence structure reaches the threshold criterion;
- `C*_dyn`: first grid point at which the dynamics structure reaches the threshold criterion;
- `C*_util`: first grid point at which utility reaches the threshold criterion.

The principal contrasts are:

\[
E_{dyn} = |C^*_{dyn} - C^*_{util}|,
\]

\[
E_{rank} = |C^*_{rank} - C^*_{util}|,
\]

\[
\Delta_{dyn,rank} = C^*_{dyn} - C^*_{rank},
\qquad
G_{sync} = |\Delta_{dyn,rank}|.
\]

Negative `Δ_dyn,rank` means that dynamics reaches its operational threshold before rank/equivalence. `sync_gap` is the absolute separation and is not itself a causal effect size.

### 2.4 What the design can and cannot establish

The design can establish whether the recorded operational markers differ under the specified training and threshold-extraction protocol. It can compare their alignment across the reported cells and describe how the grouped summaries change with residual strength.

It cannot, in its current form, establish that residual bypass causes the separation, because the current code creates a new model for each `C` and changes the seed offset as a function of `C` and residual condition. It also cannot establish a universal critical constant or generalize beyond the synthetic world without additional data and architectures.

### 2.5 Pre-specified falsifiers for the main interpretation

The phase-marker interpretation would be weakened if any of the following occurred in a preregistered re-analysis:

- fixed-model/common-random-number sweeps made the three markers effectively coincide;
- the dynamics-versus-utility alignment disappeared under neighboring grids and threshold fractions;
- residual-strength effects disappeared under a magnitude-matched shuffled-residual control;
- the effect appeared only in one split, one `K`, or one channel variant;
- the result was driven by invalid or missing thresholds rather than valid curves.

## 3. Methods

### 3.1 Synthetic symbolic bottleneck

The protocol trains a symbolic bottleneck model that maps observations into a discrete-symbol pathway and, in the residual condition, an additional continuous pathway. The nominal residual construction is:

\[
q = g(C)\,q_{sym} + r\,[1-g(C)]\,q_{cont},
\]

where `r` is residual strength. At `r = 0`, the continuous residual is removed. At `r = 1`, the full hybrid path is enabled. The `original` variant uses a linear action-logit path; the D4a `pre_softmax` variant applies a nonlinearity before the categorical logits.

The synthetic world provides state and action-related observations together with in-distribution and out-of-distribution evaluation splits. The OOD splits are not treated as independent replications of the world; they are stress tests of the same synthetic construction.

### 3.2 Threshold estimator

The protocol evaluates a discrete grid of `C` values. For each metric, it optionally smooths the interior grid points with the three-point kernel `[0.25, 0.50, 0.25]`. It then defines the operational threshold as the first grid value reaching 90% of the observed gain from the first grid point to the maximum observed value, subject to the relevant minimum-gain validity gate.

This estimator has three consequences.

1. `C*` is a grid point, not a continuous estimate.
2. The estimated threshold depends on the observed grid maximum.
3. Smoothing and the 90% fraction are part of the estimand, not merely visualization choices.

Accordingly, values such as `C* = 3.0` should be reported with the raw curve, the grid, the smoother, the gain gate, and sensitivity analyses. They should not be described as continuous critical constants.

### 3.3 D4a: decomposition and null comparison

D4a uses:

- splits: `id`, `ood_inverted`;
- variants: `original`, `pre_softmax`;
- bottleneck sizes: `K = 8, 16`;
- 20 seeds per cell;
- `C` grid: `0, 0.5, 1, 2, 3, 5, 8, 12, 16, 24`;
- formal alignment table: 8 rows, one per split × variant × `K` cell.

The comparison includes the dynamics marker, a null-dynamics marker, and the rank/equivalence marker. The integrated report summarizes the distances from each marker to utility. The machine-readable D4a alignment table additionally records q90-distance means and cell-level rates such as the fraction for which dynamics is closer to utility than rank.

### 3.4 D4b: residual desynchronization

D4b uses:

- 10 seeds;
- residual strengths `0, 0.1, 0.25, 0.5, 1.0`;
- splits `id`, `ood_random`, `ood_inverted`;
- `K = 8, 16`;
- 10 `C` values;
- 3,000 reported configurations, before grouping by strength, `K`, and split.

The grouped report records `sync_gap`, sync rate, utility sensitivity, and the negative-delta rate. The principal pattern is interpreted as an ordering change: as residual bypass is introduced, dynamics can become operationally available earlier than rank/equivalence, while the downstream utility becomes less sensitive to further movement along the `C` grid.

### 3.5 Important implementation qualification

The current D4b implementation loops over `C`, sets a seed using a function of the base seed, `K`, `C`, variant, and residual strength, constructs a new `SymbolicBottleneckModel`, and trains it for that condition. Therefore, a D4b curve is not a post-training sweep over one fixed set of weights. It is a collection of independently trained condition-specific models.

We retain this design in the current report because it is the protocol that generated the archived results. We call the resulting quantity an **operational threshold under per-`C` retraining**. The paper does not use it as evidence for a fixed-model intervention effect. A coupled or frozen sweep is a required follow-up gate.

### 3.6 Statistical unit and aggregation

Seeds and configurations are the sampling units in the archived summaries. The report-level means should not be read as if every grid point were an independent replicate. In particular, all threshold contrasts within one seed share a training condition and are statistically dependent. The next release should provide seed-level rows and confidence intervals or hierarchical uncertainty summaries rather than only grouped means.

## 4. Results

### 4.1 D4a: dynamics is the most utility-aligned tested structural marker

The archived integrated D4 report gives the following aggregate separations:

| Comparison | Mean absolute separation from `C*_util` |
| --- | ---: |
| `C*_dyn` vs. `C*_util` | 0.312 |
| null-dynamics marker vs. `C*_util` | 3.212 |
| `C*_rank` vs. `C*_util` | 12.769 |

The reported dynamics-closer-than-rank rate is 0.938. This is the strongest D4a result, but its exact interpretation is comparative: among the tested structural markers and under the operational estimator, dynamics is more utility-aligned than rank/equivalence. It does not mean that `C*_dyn` is a universal predictor of utility.

The formal D4a q90 alignment table is reproduced below from `results/d4a_repair_alignment.csv`.

| Split | Variant | K | `dyn-util` q90 error | `null-util` q90 error | `rank-util` q90 error | dyn closer rate | dyn beats null rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| id | original | 8 | 0.175 | 2.075 | 11.650 | 1.000 | 0.550 |
| id | original | 16 | 0.275 | 1.075 | 13.000 | 1.000 | 0.550 |
| id | pre_softmax | 8 | 0.325 | 6.475 | 12.850 | 0.950 | 0.650 |
| id | pre_softmax | 16 | 0.275 | 3.350 | 13.875 | 0.800 | 0.650 |
| ood_inverted | original | 8 | 0.125 | 2.025 | 11.600 | 1.000 | 0.550 |
| ood_inverted | original | 16 | 0.475 | 0.925 | 12.800 | 1.000 | 0.450 |
| ood_inverted | pre_softmax | 8 | 0.525 | 6.525 | 12.600 | 0.950 | 0.650 |
| ood_inverted | pre_softmax | 16 | 0.325 | 3.250 | 13.775 | 0.800 | 0.650 |

![D4a phase alignment](figures/d4a_phase_alignment.svg)

**Figure 1.** D4a q90 threshold error across the eight formal cells. Lower bars indicate closer alignment to utility. The figure is generated directly from `results/d4a_repair_alignment.csv`.

The D4a table supports three observations. First, all eight cells report valid detection rates for the symbolic, dynamics, null-dynamics, and rank criteria. Second, the dynamics-to-utility q90 error remains small relative to the rank-to-utility error in every cell. Third, the null dynamics comparison is worse than the learned dynamics comparison, but it is not uniformly worse under every rate-based summary. That last point is why the paper reports both distances and rates rather than compressing the null into a single binary verdict.

### 4.2 D4b: residual bypass is associated with desynchronization

The formal D4b grouped report has the following strength-level summary. These are report-level aggregates; the formal per-configuration table is a release gate, not silently reconstructed from the smoke output.

| Residual strength | Split | `sync_gap` | Sync rate | Utility sensitivity | Negative-delta rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.0 | id | 2.60 | 0.60 | 0.475 | 0.25 |
| 0.0 | ood_inverted | 3.95 | 0.65 | 0.469 | 0.25 |
| 0.0 | ood_random | 2.85 | 0.60 | 0.473 | 0.25 |
| 0.1 | id | 7.50 | 0.45 | 0.037 | 0.35 |
| 0.1 | ood_inverted | 8.95 | 0.40 | 0.212 | 0.40 |
| 0.1 | ood_random | 9.93 | 0.35 | 0.069 | 0.50 |
| 0.5 | id | 11.50 | 0.05 | 0.028 | 0.85 |
| 0.5 | ood_inverted | 10.90 | 0.05 | 0.147 | 0.85 |
| 0.5 | ood_random | 13.10 | 0.05 | 0.043 | 0.80 |
| 1.0 | id | 10.40 | 0.15 | 0.028 | 0.65 |
| 1.0 | ood_inverted | 11.70 | 0.20 | 0.143 | 0.75 |
| 1.0 | ood_random | 13.60 | 0.05 | 0.037 | 0.70 |

![D4b residual desynchronization](figures/d4b_residual_desynchronization.svg)

**Figure 2.** D4b grouped summary by residual strength and split. The blue series uses the left axis; orange and green use the right axis. This is a report-level aggregate and is not a substitute for the missing formal per-configuration CSV.

Relative to the zero-residual groups, nonzero residual groups show larger phase-marker separation and lower utility sensitivity. The negative-delta rate also becomes dominant at moderate and high residual strength, indicating that `C*_dyn < C*_rank` is more common than the reverse ordering. The pattern is not strictly monotone in every split or metric: for example, the `sync_gap` at strength 1.0 is lower than at 0.5 in the ID split. The appropriate summary is therefore “residual bypass is associated with a phase-ordering change” rather than “the effect increases monotonically with residual strength.”

### 4.3 Ordering by bottleneck size and split

The archived report gives the following mean ordering values for rank versus dynamics:

| Residual strength | K | ID | OOD random | OOD inverted |
| ---: | ---: | --- | --- | --- |
| 0.0 | 8 | 2.7 vs 2.5 | 2.7 vs 2.3 | 2.7 vs 4.6 |
| 0.0 | 16 | 6.3 vs 3.1 | 6.3 vs 3.6 | 7.1 vs 3.1 |
| 0.1 | 8 | 14.9 vs 12.5 | 14.3 vs 12.2 | 14.9 vs 12.2 |
| 0.1 | 16 | 12.8 vs 12.4 | 13.9 vs 9.6 | 14.7 vs 12.5 |
| 0.5 | 8 | 17.3 vs 7.9 | 15.6 vs 8.2 | 16.0 vs 7.9 |
| 0.5 | 16 | 23.2 vs 10.0 | 23.2 vs 5.6 | 21.2 vs 7.6 |
| 1.0 | 8 | 18.5 vs 12.0 | 16.6 vs 10.4 | 19.6 vs 12.9 |
| 1.0 | 16 | 22.0 vs 12.8 | 22.0 vs 5.9 | 22.8 vs 8.5 |

Each entry is `C*_rank vs C*_dyn`; lower dynamics values therefore represent earlier operational dynamics structure. The table shows that the direction of ordering is not a property of the split alone: it changes with residual condition and `K`. This is consistent with a phase-ordering account, but it also motivates the fixed-model and equalized-control reanalysis.

## 5. Interpretation

### 5.1 Why a single C* is insufficient here

The D4a result is best read as a failure of scalar sufficiency. The rank/equivalence marker is far from utility, while the dynamics marker is close. A single number can still be useful for a specific purpose, but it must be labeled with the phase it measures. “The critical coupling” is underspecified unless the target property is named.

### 5.2 What residual bypass may be doing

The proposed mechanism is that a continuous path carries task-relevant information without requiring the symbolic bottleneck to satisfy the same rank/equivalence constraints. Dynamics-relevant or utility-relevant behavior can therefore become operational at a lower symbolic coupling than full equivalence. The observed increase in `sync_gap`, drop in utility sensitivity, and rise in dynamics-first ordering are compatible with this account.

This is a mechanism hypothesis, not a demonstrated mediation result. The current per-`C` retraining protocol means that training randomness and optimization trajectories are part of the observed curve. A residual-strength control can change the learned solution, not just the forward path of one fixed solution. For that reason, the paper uses “associated with” and “consistent with” rather than “causes.”

### 5.3 Why utility sensitivity is not the same as utility

The D4b table reports utility sensitivity to movement along the `C` grid. A lower sensitivity can mean that utility is robust to the bottleneck control because the task is solved through a bypass; it can also mean that the model has saturated, that the threshold estimator has compressed the curve, or that training variability obscures a real effect. Sensitivity should therefore be interpreted together with absolute utility, action accuracy, threshold validity, and raw curves in the next release.

## 6. Robustness and falsification plan

The current archive provides useful but incomplete robustness evidence.

### 6.1 Already present

- D4a compares two splits, two channel variants, and two bottleneck sizes.
- D4a includes a null-dynamics comparison.
- D4b reports ID and two OOD splits separately rather than averaging them away.
- The residual-strength zero condition provides a baseline for the bypass comparison.

### 6.2 Required before a strong publication claim

#### Fixed-model or coupled-C analysis

Train once per seed and residual condition, then evaluate the same weights across the `C` grid. If the model architecture requires `C` during training, use common random numbers and a pre-specified coupling so that adjacent `C` conditions differ only in the intended control. Compare the resulting thresholds with the current per-`C` retraining estimand.

#### Common random numbers

Reuse world draws, minibatches, evaluation draws, and noise across neighboring `C` values. This reduces the risk that the apparent threshold is a change in sampled problem instances.

#### Threshold sensitivity

Report unsmoothed curves, the current smoother, alternative threshold fractions, and neighboring grids. The main direction should survive without relying on the single grid point labeled 3.0.

#### Residual nulls

Preserve residual norm and compute budget while shuffling residual directions, labels, or assignment to the symbolic path. If desynchronization remains under a semantic shuffle, the proposed bypass interpretation must be narrowed.

#### Equalized controls

Match parameter count, training steps, optimizer settings, and evaluation budget across residual strengths. If exact matching is impossible, report the imbalance and treat it as a design limitation.

#### Raw-output audit

Publish one row per seed × residual strength × `K` × `C` × split, plus the manifest and code hash. This makes it possible to recompute every grouped statistic and to detect whether missing thresholds were dropped.

### 6.3 A result that would change the paper

If the fixed-model sweep shows that the rank, dynamics, and utility markers align, the paper should be rewritten as a warning about per-`C` retraining rather than a result about residual desynchronization. If the residual shuffle reproduces the effect, the paper should retain the phase-ordering observation but remove or weaken the semantic bypass mechanism.

## 7. Limitations

1. The world is synthetic. The result is a controlled diagnostic, not evidence about natural tasks.
2. The D4b formal raw table is not present in the archived output directory, so the current public artifact is not yet fully audit-complete.
3. The per-`C` retraining design prevents a clean causal interpretation of the `C` sweep.
4. The operational threshold depends on a discrete grid, local smoothing, a 90% gain fraction, and minimum-gain gates.
5. The grouped report does not yet provide seed-level uncertainty intervals for every result.
6. The OOD splits test particular synthetic shifts and should not be interpreted as broad distributional generalization.
7. The current mechanism story is not separated from optimization, parameter-count, or compute differences.
8. The linked activation-invariance experiment is not part of the evidence base for this paper. It used a constructionally identical activation path and independent per-`C` random draws, so it should not be used to strengthen the D4 claim without a corrected protocol.

## 8. Reproducibility and artifact map

The repository keeps the manuscript separate from copied source artifacts.

| Purpose | File |
| --- | --- |
| Complete protocol and original commands | `source/D4_PROTOCOL.md` |
| D4b implementation | `source/d4b_residual_desync_gpu.py` |
| Unified entry point | `source/d4_complete_protocol.py` |
| Formal D4a alignment table | `results/d4a_repair_alignment.csv` |
| D4a/D4b report-level summary | `results/D4_integrated_report.md` |
| D4b grouped summary | `results/D4b_residual_desync_results.md` |
| Claim ceiling and release gates | `CLAIMS_AND_LIMITS.md` |
| Low-cost artifact check | `repro/validate_artifacts.ps1` |

Run the local artifact check with:

```powershell
powershell -ExecutionPolicy Bypass -File .\repro\validate_artifacts.ps1
```

This check verifies that the formal D4a table and report artifacts are present and that the D4a schema has eight rows. It intentionally reports, rather than hides, the absence of the formal D4b raw table.

## 9. Conclusion

The D4 experiment supports a narrower and more useful statement than “there is one critical coupling.” In the tested symbolic bottleneck, rank/equivalence, dynamics, and downstream utility have different operational phase markers. The dynamics marker is substantially closer to utility than the rank/equivalence marker in the archived D4a result. In the archived D4b summary, continuous residual bypass is associated with larger phase-marker separation, lower utility sensitivity, and more frequent dynamics-first ordering.

The correct next step is not to promote a particular grid point into a universal constant. It is to close the evidence gates: export the formal D4b raw table, rerun a fixed-model/common-random-number sweep, test estimator sensitivity, and add magnitude-matched residual nulls. If those checks preserve the ordering, D4 becomes a publishable controlled study of phase desynchronization. If they do not, the paper still yields a valuable methodological result: scalar C* claims can be artifacts of the training and threshold protocol unless the measured phase is specified.

## References

1. A. M. Saxe, J. L. McClelland, and S. Ganguli, “Exact solutions to the nonlinear dynamics of learning in deep linear neural networks,” arXiv:1312.6120, 2013. https://arxiv.org/abs/1312.6120
2. K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” arXiv:1512.03385, 2015; published in CVPR 2016. https://arxiv.org/abs/1512.03385
3. N. Tishby and N. Zaslavsky, “Deep learning and the information bottleneck principle,” arXiv:1503.02406, 2015. https://arxiv.org/abs/1503.02406
4. M. Raghu, J. Gilmer, J. Yosinski, and J. Sohl-Dickstein, “SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability,” arXiv:1706.05806, 2017. https://arxiv.org/abs/1706.05806

These citations provide context for learning dynamics, residual pathways, bottleneck/phase-transition language, and representation comparison. They are not evidence for the local D4 numbers; those numbers come only from the local artifacts listed in `SOURCES.md`.
