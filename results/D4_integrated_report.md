> **Historical artifact — superseded interpretation.** This report preserves an earlier D4 analysis, terminology, and mechanism-level interpretation. Its numerical tables remain archival context, but its causal or universal wording is not a current claim. Use paper.md, CLAIMS_AND_LIMITS.md, and the machine-readable raw results for the evidence-bounded account.
# Experiment D4: C* Decomposition and Residual Desynchronization

Final structure:

- D4a: decompose C* into rank/equivalence, dynamics, and utility phase markers.
- D4b: test whether continuous residual bypass desynchronizes those phase markers.

## D4a: C* Decomposition

D4a tests whether a single unsupervised C* is sufficient. The main result is that dynamics-level C* is much closer to utility C* than rank/equivalence C*, and this survives the null dynamic baseline.

- Mean |C*_dyn - C*_util|: 0.312
- Mean |C*_dyn_null - C*_util|: 3.212
- Mean |C*_rank - C*_util|: 12.769
- Mean dyn-closer-than-rank rate: 0.938

| split | variant | K | dyn-util err | null-util err | rank-util err | dyn closer | dyn beats null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| id | original | 8 | 0.175 | 2.075 | 11.650 | 1.000 | 0.550 |
| id | original | 16 | 0.275 | 1.075 | 13.000 | 1.000 | 0.550 |
| id | pre_softmax | 8 | 0.325 | 6.475 | 12.850 | 0.950 | 0.650 |
| id | pre_softmax | 16 | 0.275 | 3.350 | 13.875 | 0.800 | 0.650 |
| ood_inverted | original | 8 | 0.125 | 2.025 | 11.600 | 1.000 | 0.550 |
| ood_inverted | original | 16 | 0.475 | 0.925 | 12.800 | 1.000 | 0.450 |
| ood_inverted | pre_softmax | 8 | 0.525 | 6.525 | 12.600 | 0.950 | 0.650 |
| ood_inverted | pre_softmax | 16 | 0.325 | 3.250 | 13.775 | 0.800 | 0.650 |

## D4b: Residual Bypass Desynchronization

D4b tests the mechanism behind D4a: symbolic structure formation is synchronized when the downstream path is forced through symbols, while continuous residuals desynchronize rank/equivalence, dynamics, and utility phase markers.

Run summary:

- 3000 configurations
- 10 seeds
- 5 residual strengths
- 3 splits: `id`, `ood_random`, `ood_inverted`

Strength-level trend:

| residual strength | split | sync gap | sync rate | utility sensitivity | negative delta rate |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.0 | id | 2.600 | 0.600 | 0.475 | 0.250 |
| 0.0 | ood_inverted | 3.950 | 0.650 | 0.469 | 0.250 |
| 0.0 | ood_random | 2.850 | 0.600 | 0.473 | 0.250 |
| 0.1 | id | 7.500 | 0.450 | 0.037 | 0.350 |
| 0.1 | ood_inverted | 8.950 | 0.400 | 0.212 | 0.400 |
| 0.1 | ood_random | 9.930 | 0.350 | 0.069 | 0.500 |
| 0.5 | id | 11.500 | 0.050 | 0.028 | 0.850 |
| 0.5 | ood_inverted | 10.900 | 0.050 | 0.147 | 0.850 |
| 0.5 | ood_random | 13.100 | 0.050 | 0.043 | 0.800 |
| 1.0 | id | 10.400 | 0.150 | 0.028 | 0.650 |
| 1.0 | ood_inverted | 11.700 | 0.200 | 0.143 | 0.750 |
| 1.0 | ood_random | 13.600 | 0.050 | 0.037 | 0.700 |

Here `Delta_C_dyn_eq = C*_dyn - C*_rank`. A negative delta means dynamics emerges before rank/equivalence.

Main D4b finding:

- `sync_gap` jumps from roughly `2.6-3.95` without residuals to roughly `7.5-13.6` once residual bypass is introduced.
- `utility_sensitivity` collapses from roughly `0.47` to `0.03-0.21`.
- `negative_delta_rate` rises from `0.25` to as high as `0.80-0.85`, so dynamics-first ordering becomes dominant.

Therefore, residual bypass does not simply help or hurt utility. It changes the ordering of symbolic structure formation by allowing dynamics-relevant information to emerge at lower C while rank/equivalence constraints are delayed to higher C.

## Interpretation

The upgraded D4 should no longer claim that a single C* directly predicts downstream utility. The stronger claim is hierarchical: rank/equivalence C*, dynamics C*, and utility C* are separable; dynamics C* is the best utility-aligned marker; and residual continuous bypass changes the ordering of symbolic structure formation.
