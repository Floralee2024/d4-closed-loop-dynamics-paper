# When One Critical Coupling Is Not Enough

This repository contains the manuscript and reproducibility materials for the D4 experiment:

> **When One Critical Coupling Is Not Enough: Residual Bypass Desynchronizes Representation, Dynamics, and Utility**

The paper studies a synthetic symbolic bottleneck with three operational phase markers:

- `C*_rank`: rank/equivalence structure,
- `C*_dyn`: dynamics-level structure,
- `C*_util`: downstream utility.

The central result is hierarchical rather than scalar: the three markers are separable, `C*_dyn` is more closely aligned with `C*_util` than the rank/equivalence marker in the current design, and a continuous residual bypass changes their ordering instead of producing a simple “better/worse” effect on utility.

## Status

`paper.md` is a complete evidence-bounded manuscript draft. It is intentionally explicit about two publication gates that are not yet closed:

1. the formal D4b run is currently preserved as report-level grouped results; the formal per-configuration CSV was not found in the source output directory;
2. the current D4b implementation retrains a fresh model for every `C`, so the reported curves are operational thresholds under per-`C` retraining, not a fixed-trained-model intervention sweep.

These are not cosmetic caveats. The first limits auditability; the second limits the causal interpretation of a `C` curve. The manuscript therefore does not describe `C*` as a continuous critical constant or claim that residual bypass alone caused the observed phase ordering.

## Repository layout

```text
paper.md                         manuscript draft
CLAIMS_AND_LIMITS.md             claim ceiling and release gates
SOURCES.md                       provenance of local artifacts
results/d4a_repair_alignment.csv formal D4a alignment table
results/D4_integrated_report.md  source D4 summary
results/D4b_residual_desync_results.md
source/                          copied experiment entry points
  d4_complete_protocol.py        unified entry point
  d2_d4_protocol.py              D4a implementation
  experiment_D_pytorch_history_symbol_budget.py
                                  D4a model dependency
  d4b_residual_desync_gpu.py     D4b implementation
  summarize_d4.py                report generator
repro/                           low-cost validation and rerun notes
```

## Reproduction entry points

The source protocol contains the original smoke and full-run commands. From the repository root, the intended commands are:

```powershell
python .\source\d4_complete_protocol.py --stage smoke --device auto
python .\source\d4_complete_protocol.py --stage all --device cuda --tf32 --persistent-workers
```

The original full D4b command is recorded in `source/D4_PROTOCOL.md`. It is expensive and should only be run after the fixed-model sweep and raw-output export gates described in `CLAIMS_AND_LIMITS.md` are addressed. The smoke command is a command-chain check; it is not a substitute for the formal result.

The local artifact check also runs `repro/audit_claim_numbers.py`, which recomputes the headline D4a means and checks the D4b report-level table shape and separation ranges.

## Current interpretation

The strongest defensible statement from the archived evidence is:

> In this synthetic symbolic-bottleneck protocol, representation geometry, dynamics structure, and downstream utility do not share one operational threshold. Under the reported per-`C` retraining protocol, the dynamics marker is closer to utility than the rank/equivalence marker, while residual bypass is associated with larger phase-marker separation and lower measured utility sensitivity.

The result is a mechanism hypothesis and a reproducible benchmark observation, not yet a universal law about neural networks.
