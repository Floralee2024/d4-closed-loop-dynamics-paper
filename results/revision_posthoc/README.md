# Post-hoc revision analysis

Status: exploratory descriptive sensitivity analysis, 26 September 2026. No training or archived result files were changed.

Ranges are computed within seed x K before equal weighting of K-specific seed means. Positive-C analysis excludes only C=0; it does not redefine or recompute the archived thresholds.

| r | Split | U(0) | U(24) | Full-grid range | C>0 range |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | id | 0.000000 | 0.444846 | 0.469560 | 0.064604 |
| 0 | ood_random | 0.000000 | 0.456332 | 0.476776 | 0.064460 |
| 0 | ood_inverted | 0.000000 | 0.430397 | 0.465948 | 0.087950 |
| 0.1 | id | 0.525950 | 0.509294 | 0.030391 | 0.029788 |
| 0.1 | ood_random | 0.382991 | 0.426310 | 0.057882 | 0.056744 |
| 0.1 | ood_inverted | 0.187272 | 0.306555 | 0.185809 | 0.181746 |
| 0.25 | id | 0.526299 | 0.521334 | 0.024797 | 0.024679 |
| 0.25 | ood_random | 0.379211 | 0.400939 | 0.041652 | 0.040028 |
| 0.25 | ood_inverted | 0.182002 | 0.234612 | 0.139841 | 0.126132 |
| 0.5 | id | 0.527041 | 0.521258 | 0.025613 | 0.024015 |
| 0.5 | ood_random | 0.379210 | 0.396326 | 0.039166 | 0.038363 |
| 0.5 | ood_inverted | 0.188693 | 0.234734 | 0.154193 | 0.150467 |
| 1 | id | 0.525454 | 0.524063 | 0.020948 | 0.019963 |
| 1 | ood_random | 0.381614 | 0.391054 | 0.035749 | 0.034811 |
| 1 | ood_inverted | 0.200924 | 0.212917 | 0.150641 | 0.148380 |

`denominators.csv` preserves the archived all-row event rates and adds explicitly labeled pooled-valid rates; these do not silently replace the K-balanced published estimand. Empty thresholds are not evidence for the opposite ordering.

`bootstrap_contrasts.csv` reports post-hoc 95% percentile intervals from 2,000 base-seed resamples (seed 20260926), jointly retaining both K levels and both residual conditions. Intervals condition on this world, grid and valid-threshold rule; they do not correct censoring, missingness or multiple comparisons and are not confirmatory significance tests.

D4a artifacts were subsequently recovered from the recorded SOURCES.md archive path. See results/d4a_formal_archive and repro/verify_d4a_archive.py. The D4b analyses here do not use or reconstruct D4a data.
