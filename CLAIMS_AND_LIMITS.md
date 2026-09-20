# Claims, evidence, and release gates

This file prevents the repository from silently upgrading a benchmark result into a stronger scientific claim.

## Evidence levels

| Level | Supported by the current repository? | Meaning |
| --- | --- | --- |
| Protocol/data integrity | Partly | D4a has a machine-readable formal alignment table; D4b has report-level grouped results and a copied implementation, but the formal raw D4b table is not in the archived output directory. |
| Association | Yes, conditionally | Across the reported D4a cells, `C*_dyn` is closer to `C*_util` than `C*_rank`; across the reported D4b groups, residual strength is associated with larger separation and lower utility sensitivity. |
| Intervention effect | Not yet | The current D4b implementation changes residual strength, but also retrains independently at each `C` and changes the seed offset by residual condition and `C`. A fixed-model/common-random-number sweep is required before assigning a clean intervention effect to the residual path. |
| Mechanism | Not yet | The bypass interpretation is compatible with the results, but a shuffled-residual null, fixed-model sweep, and raw per-configuration audit are still required to distinguish the mechanism from protocol-induced variation. |
| Generalization | No | The evidence is from one synthetic family and a limited set of splits, widths, variants, and seeds. |

## Claims the paper makes

1. A single operational `C*` is insufficient to summarize all three measured phases in this experiment.
2. In the D4a archive, the dynamics marker is closer to utility than the rank/equivalence marker by the reported q90-distance summary.
3. In the D4b archive, residual-bypass conditions are accompanied by larger `|C*_dyn - C*_rank|`, lower utility sensitivity, and a higher rate of dynamics-first ordering.
4. `C*` is a grid-derived operational threshold: the first grid point reaching 90% of the observed gain after the specified smoothing and validity gate.

## Claims the paper does not make

- `C* = 3.0` is not a continuous critical constant.
- The results do not show that one C* predicts downstream utility in general.
- The results do not establish a causal mechanism under the current per-C retraining protocol.
- The results do not establish that residual bypass universally improves or harms utility.
- The results do not establish transfer to natural data, other architectures, or deployment settings.

## Required gates before calling the repository submission-ready

### Gate A: export the formal D4b raw table

The formal run should emit, at minimum, one row per `(seed, residual_strength, variant, K, C, split)` with the raw metrics used to compute the C* markers. The output must include a configuration manifest, code version, device, and completion count. The existing smoke CSV must not be merged into the formal table.

### Gate B: fixed-model/common-random-number C sweep

For each seed and residual condition, train the model once under a pre-specified training coupling, freeze the resulting weights, and evaluate the same model over the C grid. If training genuinely depends on C by design, report this as a separate estimand and add a frozen or coupled alternative. Reuse the same worlds, batches, and evaluation draws across C values.

The repository includes `repro/fixed_model_sweep.py` for an explicit diagnostic version of this gate. The first 16-cell probe completed, but every cell had `dyn_gain=0`: the current dynamics score is based on hard symbol IDs, which are C-invariant once weights are frozen. Therefore the naive fixed-model sweep is not a valid test of the current `C*_dyn` estimand. A valid follow-up must first define a C-dependent dynamics metric or explicitly treat `C*_dyn` as a training-path threshold. The probe and raw outputs are archived in `results/fixed_model_sweep_probe/`.

### Gate C: estimator sensitivity

Recompute all phase markers under at least:

- raw unsmoothed curves,
- the current `[0.25, 0.50, 0.25]` interior smoother,
- neighboring C grids or a prespecified interpolation rule,
- q90 thresholds such as 0.80 and 0.95, reported as sensitivity rather than selected after seeing the result.

### Gate D: mechanism falsification

Add a residual-direction or residual-label shuffle that preserves magnitude and training budget but breaks the proposed semantic bypass. Add an equalized-parameter or equalized-compute control if possible. The mechanism claim should survive these controls or be narrowed.

### Gate E: release audit

Run `repro/validate_artifacts.ps1`, inspect the diff, record hashes for scripts and result tables, and only then create the public GitHub repository.
