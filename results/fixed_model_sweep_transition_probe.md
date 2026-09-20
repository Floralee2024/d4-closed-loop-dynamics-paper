# Fixed-model probe with a C-dependent dynamics readout

This probe repeats the fixed-model design using the transition-head gain as the dynamics score. Unlike hard-ID NMI, the transition head consumes `q(C)`, so the score can change with C after weights are frozen.

Configuration:

- 2 seeds, `K=8`, `original` variant;
- residual strengths `0, 0.1, 0.5, 1.0`;
- `id` and `ood_inverted` splits;
- `C = 0, 0.5, 1, 2, 3, 5, 8`;
- one training run per seed × residual condition at `train_C=8`;
- 200 training steps, 512 training sequences, 256 evaluation sequences.

All 16 cells produced a valid `C*_dyn` under the transition-head metric. Mean `sync_gap` by residual strength and split was:

| residual strength | id | ood_inverted |
| ---: | ---: | ---: |
| 0.0 | 0.5 | 0.5 |
| 0.1 | 3.5 | 1.5 |
| 0.5 | 1.5 | 2.5 |
| 1.0 | 0.0 | 4.0 |

The probe shows that a fixed-model C-dependent dynamics metric is technically well-defined, but the pattern is non-monotone and several ID cells have no valid `C*_util` under the utility gain gate. With two seeds, one K, one training C, and no residual shuffle/equalized control, this is a method diagnostic rather than a formal intervention result. The raw artifacts are `config.json`, `curves.csv`, and `cstar_summary.csv` in this directory.
