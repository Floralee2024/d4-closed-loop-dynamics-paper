# Supervised Criterion Selection and Held-Out Utility in a Synthetic Bottleneck Task

**Draft status:** research manuscript draft, 26 October 2026

**Author:** Flora Lee (sole author)

**Affiliation:** Independent researcher and software developer (no institutional affiliation)

## Abstract

Measurement studies can show that internal readouts track task scores without establishing whether changing a training objective changes task performance. We report a separate supervised intervention that follows the D4b per-control retraining protocol. At each control value and residual strength, a symbolic-bottleneck model was freshly trained with either no additional criterion loss or one of several criterion-weighting procedures. The primary comparison used 40 independent base seeds and hidden OOD-inverted task utility. A validation-only selector chose one executable three-weight criterion for each seed and reused it across target cells. The selected arm, joint_oracle_filtered, exceeded the zero-additional-criterion arm, joint_none, by 0.005711 normalized utility points on average. A two-sided Student-t interval recomputed from the archived seed-level paired contrasts was [0.001736, 0.009687]; 28 of 40 seed-level contrasts were positive. The archived machine record reports PASS and NO_RL_PILOT_PASSED_RL_MAY_BE_PLANNED. The internal readouts did not all move upward with utility: the mean effective-rank change was -0.016888, and the independent q-space horizon-probe change was +0.000681. These results support an effect of the validation-selected supervised training procedure on this synthetic held-out prediction score. They do not identify which criterion component produced the difference, establish a general principle-reasoning ability, or demonstrate transfer to natural-language models or real-world tasks.

## 1. Introduction

The preceding D4 studies compared operational landmarks for representation, temporal association, and task utility. D4a trained one model per seed and representation variant while changing the data-generating control parameter. D4b retrained models at each control value, residual strength, and bottleneck size. Their measures, utility scales, threshold procedures, and training protocols differ, so the resulting q90 landmarks are protocol-specific summaries [1]. Within those archived comparisons, D4a's selected temporal-association q90 marker was closer to the utility marker than the selected use/rank marker; on D4b's original grid, nonzero residual conditions were associated with greater marker spacing. These are discrete grid-marker comparisons, not estimates of a shared continuous onset.

A later curve review preserved the historical measurements while narrowing the interpretation. Recalculation matched all 1,860 stored landmarks, while also finding marks at grid boundaries, curves that fell below threshold again, and undefined landmarks. In D4a, 140 of 160 utility marks were at C=0. Their proximity to another mark can therefore reflect a shared boundary value rather than synchronized ability onset. We treat the D4 results as motivation for a distinct question: when a training criterion is actively selected and added during supervised learning, does held-out task utility change?

This study is the direct supervised follow-up. It retains the D4b per-C retraining estimand and introduces a seven-arm intervention over additional training criteria and their selection sources. The primary contrast is a validation-selected criterion package against a control with zero additional criterion loss. This comparison tests a training procedure as a whole. It does not test whether one representation metric is a sufficient statistic for task performance, nor does it isolate a causal path through a particular internal readout. Questions about proxy objectives and their relation to desired outcomes have a broader history in machine learning safety research [2].

## 2. Experiment card and estimand

**Question.** Under the fixed synthetic sequence task and D4b-style per-C retraining, does a validation-selected supervised criterion-weighting procedure improve held-out OOD-inverted task utility relative to ordinary task training with no additional criterion loss?

**Intervention.** Select one three-weight criterion from a fixed candidate bank using validation utility at a single anchor condition, then add the selected criterion losses to supervised training at all target cells.

**Primary outcome.** The hidden task utility defined by the runner as majority-adjusted action-classification accuracy, averaged over positive-C and residual-strength cells.

**Independent unit.** Base seed. Each seed contributes one paired arm contrast after averaging its repeated target cells. Cells, examples, sequence positions, and fitted selector runs are not treated as independent experimental units.

**Supported inference.** An effect of the complete validation-selection-plus-supervised-training procedure on the specified synthetic held-out score, conditional on this model, task, grid, and implementation.

**Not established.** A mechanism through rank or dynamics proxies; universal principle reasoning; action-conditioned control; performance on a real-world or language-model task; or a fixed-model forward-path intervention effect.

## 3. Methods

### 3.1 Synthetic task and training protocol

The experiment used the discrete symbolic-bottleneck model and the residual-bypass family from the D4b codebase. The formal run used K=8 symbols, an observation dimension of 32, a hidden dimension of 64, a 12-dimensional embedding, and the original model variant. Sequence length was 8. A fresh model was trained for every arm, base seed, C value, and residual-strength condition. Thus, the estimand concerns retraining outcomes under each cell's training procedure; it is not a weight-frozen intervention.

The training grid contained C in {0, 1, 3, 8} and residual strength in {0, 0.5, 1}. The primary utility aggregation excluded C=0 and averaged the three positive-C values and three residual strengths within each base seed. C=0 remained in the released diagnostic curves. The primary evaluation split was ood_inverted; training and validation used the mixed synthetic distribution. The hidden split was excluded from model fitting and criterion selection. The formal run used 40 base seeds, 20 through 59, 60 training steps per cell, batches of 128, and 512 training and validation sequences per cell. The optimizer was AdamW with learning rate 0.002 and weight decay 0.0001.

Utility was the normalized excess action-prediction accuracy

\[
U = \frac{A - A_{majority}}{1-A_{majority}},
\]

where A is action-label accuracy and A_majority is the accuracy of the majority action class on the evaluation data. This is a held-out prediction score on a synthetic task. It is not an episode return from an interactive environment.

### 3.2 Arms and training criteria

The baseline supervised objective included action and future-outcome prediction, symbol-usage and conditional-entropy regularization, and a symbol-transition loss. The additional criterion terms were: (i) a latent-code equivalence-contrastive rank term, (ii) an additional future-outcome prediction term, and (iii) an additional action-prediction term. The rank term uses the known synthetic latent code as an oracle-assisted structural target. The future-outcome term is supervised prediction of an observed target, not an online transition intervention.

The seven arms were:

| Arm | Additional criterion rule |
|---|---|
| joint_none | All three additional criterion coefficients are zero; ordinary task training remains active. |
| joint_default | Unfiltered default weights: rank 0.20, dynamics 0.50, utility 0.50. |
| joint_fixed | Fixed analytic-balanced weights: 0.40, 0.40, 0.40. |
| joint_self_unfiltered | A proposal chosen without validation-utility filtering. |
| joint_oracle_filtered | Candidate with the largest validation-utility lower confidence bound. |
| joint_misaligned | Candidate selected to maximize the validation rank-plus-dynamics proxy score. |
| joint_random | Random candidate from the same fixed bank. |

The additional loss was L_add = w_rank L_rank + w_dynamics L_future + w_utility L_action. The common supervised objective remained active in every arm. Thus joint_none is a zero-additional-loss reference, not a model trained without task supervision.

Each candidate used nonnegative rank, dynamics, and utility weights from a four-item bank: (0.40, 0.40, 0.40), (0.60, 0.30, 0.30), (0.30, 0.60, 0.30), or (0.30, 0.30, 0.60). Each candidate had the same L1 weight budget of 1.20. The validation-filtered candidate was chosen once per base seed at the anchor K=8, residual strength 0.5, original variant, and C=3. Candidate selectors used 12 training steps, two selector repeats, and three validation replicas. Selection maximized mean validation utility minus one standard error. The selected candidate was reused across all target cells for that seed. The hidden split was not supplied to this procedure.

The dynamics readout was an independent q-space predictive probe for one- and two-step future outcomes, trained on the training split and scored on each evaluation split. It measures predictive association. It does not measure whether an agent's action changes the environment's future state.

### 3.3 Statistical analysis

The primary contrast was joint_oracle_filtered minus joint_none. For each seed, utility was first averaged over C in {1, 3, 8} and residual strength in {0, 0.5, 1}; the paired seed differences were then summarized across 40 seeds. The reported confidence interval is a two-sided 95% Student-t interval with 39 degrees of freedom, recomputed from the archived seed-level paired values in seed_level_effects.csv. The redesign power plan used a smallest effect size of interest (SESOI) of 0.003 to choose the seed count (37 estimated, rounded to 40). The formal promotion gate encoded a positive mean contrast (target_delta > 0), not a test that the effect exceeded the SESOI. The archived PASS must therefore be read against its encoded positive-effect rule; the confidence interval is reported separately as a summary of the seed-level values.

## 4. Results

### 4.1 Hidden utility

The validation-filtered arm had a mean seed-level utility difference of +0.005711 versus joint_none (n=40 base seeds; 95% t interval [+0.001736, +0.009687]). Twenty-eight of the 40 seed-level differences were positive. The archived gate record reports pilot_pass=true, hidden_utility_pass=true, anti_hacking_pass=true, and the decision NO_RL_PILOT_PASSED_RL_MAY_BE_PLANNED. The point estimate exceeds the planned SESOI of 0.003, but the 95% interval lower endpoint (+0.001736) does not; these data do not establish with 95% confidence that the effect is at least 0.003. The record also has a separate all-arm row-level no_proxy_only_risk sensitivity flag set to false (151 operational-proxy rows, 90 raw-proxy rows, including 10 raw-proxy rows in the target arm). That sensitivity flag was explicitly excluded from the promotion checks; the required aggregate target_no_systematic_proxy_only check passed. The PASS is not evidence that local proxy-only cases were absent.

The machine record documents that the required promotion checks passed, while also retaining the separate failed row-level sensitivity flag described above. It is not evidence that every selected readout improved. The target arm's mean normalized effective-rank (erank_norm) difference versus joint_none was -0.016888. Its mean q-probe horizon-gain difference was +0.000681. The utility gain therefore co-occurred with a lower normalized effective-rank readout on average and a near-zero mean change in the predictive dynamics readout. No claim that the intervention improved all internal metrics is supported.

### 4.2 Interpretation of the intervention

The result supports a positive average effect for the validation-selected supervised procedure under this synthetic task and evaluation split. The primary treatment includes candidate training, validation-based selection, coefficient choice, and the additional losses used in final target-cell training. Because the intervention changes several linked pieces, this study cannot assign the utility difference to the rank, dynamics, or utility loss separately. The word oracle in the arm name refers to the predefined validation-based selection rule. It does not indicate hidden-test access or general reasoning ability.

## 5. Discussion

The study moves the D4 research program from comparing measurements to changing a training procedure. Its main finding is that, in this specified supervised synthetic setting, validation selection over a small executable criterion bank produced a small positive average change in a held-out task-prediction score. The effect is an intervention result at the training-procedure level. It does not show that a metric improvement itself causes a utility increase. Indeed, the rank proxy decreased on average while the task score increased.

Three design choices define the scope. First, the control is genuinely zero in additional rank, dynamics, and utility criterion losses, while retaining the task's ordinary supervised losses. Second, the selection decision is made at a single validation anchor and reused across target cells. Third, utility is evaluated on an inverted-spurious-feature split withheld from selection. These choices help define a clean procedure-level contrast while leaving its internal pathway unresolved.

Several limitations remain. The utility is a normalized classification score, not reward from a closed-loop agent. The world, labels, data distributions, and model are synthetic. The candidate bank is small and hand-specified. The rank criterion has access to synthetic latent-code labels. The outcome does not establish that a natural-language model can represent or reason with general principles. Finally, fresh retraining at each C and residual value answers a different question from holding weights fixed and changing only a forward-path component.

The next test most likely to change the interpretation would separate selection from criterion content: compare the same selected coefficients against matched unfiltered and misaligned coefficients under a design with a predeclared primary utility contrast and additional held-out seeds. A separate fixed-model study would be required to test whether a change in representation readout mediates the utility result.

## 6. Data, code, and provenance

The local run bundle used for this draft is under **outputs/principled_provenance_v4_confirm_40seed_cpu_20261008/**. It contains config.json, anti_hacking_gate.json, seed_level_effects.csv, arm_comparisons.csv, arm_performance.csv, curves.csv, principle_selection.csv, and the selection manifest. The predecessor-curve claims above are documented in the local D4a/D4b audit at **results/cstar_curve_shape_audit/curve_audit_report.md**; that audit recomputed archived curves and did not rerun training. The training runner is **repro/cstar_principled_pilot.py**; it imports the D4b model and data implementation from **source/d4b_residual_desync_gpu.py**.

The local repository's .gitignore excludes outputs/. The formal run bundle and the runner were therefore not part of the current public repository snapshot when this draft was prepared. A tracked, versioned artifact release is pending; the manuscript should not be described as fully reproducible from the public repository until the relevant configuration, seed-level data, gate record, and code are published together.

## 7. Author contributions and AI assistance

Flora Lee is the sole author and conducted this work as an independent researcher and software developer. She formulated the research questions, designed and directed the studies, interpreted the results, reviewed anomalous outcomes, and prepared and approved the manuscript. AI-assisted tools supported model and code implementation, training execution, comparison, aggregation, and analysis under her direction. The tools are acknowledged as tools and are not authors; Flora Lee takes responsibility for the final manuscript and its claims.

## References

1. D4 project manuscript, “When One Critical Coupling Is Not Enough: Residual Bypass Is Associated with Operational Marker Separation.” https://github.com/Floralee2024/d4-closed-loop-dynamics-paper/blob/master/paper.md
2. D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, “Concrete Problems in AI Safety,” 2016. https://arxiv.org/abs/1606.06565
