# Fixed-model probe: diagnostic result

This probe was run from `repro/fixed_model_sweep.py` with:

- 2 seeds;
- `K=8`;
- residual strengths `0, 0.1, 0.5, 1.0`;
- splits `id`, `ood_inverted`;
- `C = 0, 0.5, 1, 2, 3, 5, 8`;
- 200 training steps at `train_C=8`;
- CPU PyTorch 2.14.0+cpu;
- 512 training sequences and 256 evaluation sequences per seed.

The run completed and produced 16 seed × residual × split summaries. All 16 summaries had `dyn_gain=0` and therefore no valid `C*_dyn` under the current gate.

This is an implementation-level diagnostic, not evidence that fixed-model dynamics is absent. In the current D4b metric, the dynamics score is computed from NMI of hard symbol IDs. The symbol IDs come from the model logits and do not depend on `C` when weights are frozen. Consequently, a naive fixed-weight C sweep cannot identify a C-dependent `C*_dyn` for this metric.

The result changes the follow-up design: before using a fixed-model sweep as a causal check, define a dynamics readout that actually changes with `C` (for example, a q-space or transition-head metric), or state explicitly that the current `C*_dyn` estimand is a training-path threshold. The raw artifacts are `config.json`, `curves.csv`, and `cstar_summary.csv` in this directory.
