# When One Critical Coupling Is Not Enough

The manuscript is **When One Critical Coupling Is Not Enough: Residual Bypass Is Associated with Operational Marker Separation**.

This repository contains two related but different synthetic studies:

- **D4a:** a recurrent model trained once per seed/variant; C changes sequence generation. A usage/rank proxy and an entropy/future-MI proxy are compared with normalized MSE utility using unsmoothed q90 landmarks.
- **D4b:** a discrete-bottleneck model retrained at every C, residual strength and K. A mixed-representation rank proxy and an outcome/lower-state NMI association proxy are compared using smoothed, first-point-referenced q90 landmarks.

D4b does not demonstrate the mechanism behind D4a. Their control parameters, proxies and utility definitions differ. The observations are protocol-specific associations, not fixed-model intervention effects or demonstrated symbolic equivalence.

## Current findings and qualifications

D4a's selected temporal proxy is closer to utility than its usage/rank proxy under q90. It is not the best of every diagnostic or an estimator-independent utility predictor.

D4b nonzero residual conditions have larger K-balanced marker gaps and smaller utility ranges on the original full grid. The latter contrast includes q(0,0)=0. A post-hoc analysis excluding C=0 reverses the range comparison in OOD-inverted at r=0.1. High-residual rank markers frequently hit C=24; many utility thresholds are invalid. K-balanced means and event-rate denominators are defined explicitly in the manuscript.

## Artifacts and reproduction

- `paper.md`: revised manuscript, including separate methods and post-hoc sensitivity results.
- `CLAIMS_AND_LIMITS.md`: current claim boundaries.
- `source/`: original implementations and archived protocol.
- `results/d4a_repair_alignment.csv`: eight-cell D4a aggregate table. Matching raw curves and landmarks have now been recovered in `results/d4a_formal_archive/`, with source hashes.
- `results/d4b_formal_gpu/`: 3,000 raw evaluation rows, thresholds, configs and provenance (1,000 trained conditions evaluated on three splits).
- `results/revision_posthoc/`: exploratory endpoint sensitivity, denominator diagnostics, cluster-bootstrap contrasts and input hashes.
- `figures/`: original summary figures and additional utility/proxy curves.

Run without training:

```powershell
python .\repro\audit_claim_numbers.py
python .\repro\verify_d4a_archive.py
python .\repro\revision_posthoc.py
```

The revision script uses Python's standard library and preserves archived results. Its bootstrap intervals are post-hoc descriptive summaries, not confirmatory tests. All revision tables are computed from existing CSVs; no new GPU run is needed.

Original training entry points remain available:

```powershell
python .\source\d4_complete_protocol.py --stage smoke --device auto
python .\source\d4_complete_protocol.py --stage all --device cuda --tf32 --persistent-workers
```

Historical reports and `source/D4_PROTOCOL.md` preserve earlier terminology and commands. The current manuscript supersedes their causal or universal interpretations. The two archived fixed-model probes use different readouts and remain exploratory.

D4a archive recovery reproduced 960 q90 markers. Of 160 utility markers, 140 occur at C=0; alignment is not evidence of coincident capability onset. The minimum recorded normalized symbol-usage entropy is 0.755, so complete symbol collapse is not supported by these curves. Figure 2 shows the recovered raw metric trajectories.
