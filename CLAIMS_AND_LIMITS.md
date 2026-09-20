# Claims, evidence, and release gates

This file prevents the repository from silently upgrading a benchmark result into a stronger scientific claim.

## Evidence levels

| Level | Supported by the current repository? | Meaning |
| --- | --- | --- |
| Protocol/data integrity | Yes for the per-C retraining estimand | D4a has a machine-readable formal alignment table; D4b now has 3,000 raw rows, four shard configs, a merge manifest, provenance, and an independent local re-merge check. |
| Association | Yes, conditionally | Across the reported D4a cells, `C*_dyn` is closer to `C*_util` than `C*_rank`; in formal D4b per-C retraining, residual strength is associated with larger separation and lower utility sensitivity. Nonzero conditions often have invalid `C*_util` thresholds, which limits three-marker comparisons. |
| Intervention effect | Not yet | The current D4b implementation changes residual strength, but also retrains independently at each `C` and changes the seed offset by residual condition and `C`. A fixed-model/common-random-number sweep is required before assigning a clean intervention effect to the residual path. |
| Mechanism | Not yet | The bypass interpretation is compatible with the results, but a shuffled-residual null and fixed-model sweep are still required to distinguish the mechanism from protocol-induced variation. |
| Generalization | No | The evidence is from one synthetic family and a limited set of splits, widths, variants, and seeds. |

## Claims the paper makes

1. A single operational `C*` is insufficient to summarize all three measured phases in this experiment.
2. In the D4a archive, the dynamics marker is closer to utility than the rank/equivalence marker by the reported q90-distance summary.
3. In the formal D4b per-C retraining estimand, residual-bypass conditions are accompanied by larger `|C*_dyn - C*_rank|`, lower utility sensitivity, and a higher rate of dynamics-first ordering, with validity counts reported.
4. `C*` is a grid-derived operational threshold: the first grid point reaching 90% of the observed gain after the specified smoothing and validity gate.

## Claims the paper does not make

- `C* = 3.0` is not a continuous critical constant.
- The results do not show that one C* predicts downstream utility in general.
- The results do not establish a causal mechanism under the current per-C retraining protocol.
- The results do not establish that residual bypass universally improves or harms utility.
- The results do not establish transfer to natural data, other architectures, or deployment settings.

## Required gates before calling the repository submission-ready

### Gate A: export the formal D4b raw table — closed for the current estimand

`results/d4b_formal_gpu/` now contains one row per `(seed, residual_strength, variant, K, C, split)`, a configuration manifest, four shard configs, provenance, device/runtime details, and the recomputed summaries. The local merge check reports 3,000 unique conditions and rejects missing or duplicate rows. The existing smoke CSV was not merged. This closes the raw-export gate but does not change the per-C retraining estimand.

### Gate B: fixed-model/common-random-number C sweep

For each seed and residual condition, train the model once under a pre-specified training coupling, freeze the resulting weights, and evaluate the same model over the C grid. If training genuinely depends on C by design, report this as a separate estimand and add a frozen or coupled alternative. Reuse the same worlds, batches, and evaluation draws across C values.

The repository includes `repro/fixed_model_sweep.py` for an explicit diagnostic version of this gate. The first hard-ID probe completed, but every cell had `dyn_gain=0`: hard symbol IDs are C-invariant once weights are frozen. A second 16-cell probe using the C-dependent transition-head gain produced valid `C*_dyn` values in all cells; its mean `sync_gap` ranged from 0.0 to 4.0 across residual/split groups. This partially closes the metric-definition issue, but it is not a formal intervention result: it has two seeds, one K, one training C, no residual shuffle, and several invalid `C*_util` cells. Both probes and raw outputs are archived under `results/fixed_model_sweep_probe/` and `results/fixed_model_sweep_transition_probe/`.

### Gate C: estimator sensitivity

Recompute all phase markers under at least:

- raw unsmoothed curves,
- the current `[0.25, 0.50, 0.25]` interior smoother,
- neighboring C grids or a prespecified interpolation rule,
- q90 thresholds such as 0.80 and 0.95, reported as sensitivity rather than selected after seeing the result.

### Gate D: mechanism falsification

Add a residual-direction or residual-label shuffle that preserves magnitude and training budget but breaks the proposed semantic bypass. Add an equalized-parameter or equalized-compute control if possible. The mechanism claim should survive these controls or be narrowed.

### Gate E: release audit

Run `repro/validate_artifacts.ps1`, inspect the diff, record hashes for scripts and result tables, and keep the GitHub release aligned with the verified local state. The repository is currently private; making it public is a separate authorization decision.
