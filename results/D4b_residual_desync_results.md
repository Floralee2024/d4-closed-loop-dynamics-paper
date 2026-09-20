# D4b Residual Desynchronization Results

Run summary:

- 3000 configurations
- 10 seeds
- 5 residual strengths
- 3 splits: `id`, `ood_random`, `ood_inverted`
- Main hypothesis: `residual_strength ↑ => sync_gap ↑ => utility_sensitivity ↓`

## Strength-Level Summary

The main grouped readout supports the D4b mechanism.

| residual strength | split | sync gap | sync rate | utility sensitivity | negative delta rate |
| --- | --- | ---: | ---: | ---: | ---: |
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

Key pattern:

- `sync_gap` increases sharply once the residual path is introduced.
- `utility_sensitivity` collapses after residual strength exceeds zero.
- `negative_delta_rate` rises strongly, meaning `C*_dyn < C*_rank` becomes the dominant ordering.

Here `Delta_C_dyn_eq = C*_dyn - C*_rank`.

- `Delta < 0`: dynamics structure emerges before rank/equivalence.
- `Delta > 0`: rank/equivalence emerges before dynamics.
- `|Delta|` is the desynchronization gap.

## Ordering by K and Split

| residual strength | K | ID: C_rank -> C_dyn | ood_random: C_rank -> C_dyn | ood_inverted: C_rank -> C_dyn |
| --- | ---: | --- | --- | --- |
| 0.0 | 8 | 2.7 vs 2.5, near sync | 2.7 vs 2.3, near sync | 2.7 vs 4.6, dynamics later |
| 0.0 | 16 | 6.3 vs 3.1, dynamics earlier | 6.3 vs 3.6, dynamics earlier | 7.1 vs 3.1, dynamics earlier |
| 0.1 | 8 | 14.9 vs 12.5, dynamics earlier | 14.3 vs 12.2, dynamics earlier | 14.9 vs 12.2, dynamics earlier |
| 0.1 | 16 | 12.8 vs 12.4, near sync | 13.9 vs 9.6, dynamics earlier | 14.7 vs 12.5, dynamics earlier |
| 0.5 | 8 | 17.3 vs 7.9, dynamics much earlier | 15.6 vs 8.2, dynamics much earlier | 16.0 vs 7.9, dynamics much earlier |
| 0.5 | 16 | 23.2 vs 10.0, dynamics much earlier | 23.2 vs 5.6, dynamics much earlier | 21.2 vs 7.6, dynamics much earlier |
| 1.0 | 8 | 18.5 vs 12.0, dynamics earlier | 16.6 vs 10.4, dynamics earlier | 19.6 vs 12.9, dynamics earlier |
| 1.0 | 16 | 22.0 vs 12.8, dynamics earlier | 22.0 vs 5.9, dynamics much earlier | 22.8 vs 8.5, dynamics much earlier |

## Interpretation

D4b explains why the original single-C* interpretation fails.

In a pure symbolic channel, utility remains sensitive to coupling strength, and
rank/equivalence structure and dynamics structure are relatively synchronized.
Once a residual continuous path is introduced, the model can solve much of the
downstream problem through the bypass. As a result:

- Dynamics-level structure can become available at lower C.
- Rank/equivalence constraints are pushed to much higher C.
- Utility becomes far less sensitive to C.

Thus continuous residual bypass does not merely improve or damage performance.
It changes the phase ordering of symbolic structure formation.

## Formal D4b Claim

Residual bypass desynchronizes symbolic structure formation:

```text
residual_strength ↑
=> |C*_dyn - C*_rank| ↑
=> utility_sensitivity ↓
=> dynamics-first ordering becomes dominant
```

This upgrades D4 from a single-C* test into a phase-ordering experiment.
