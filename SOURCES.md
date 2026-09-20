# Source provenance

The files in this repository are copied from the local D4 experiment archive. The copied files are kept separate from the manuscript so that prose changes do not modify the original experiment outputs.

## Local source paths

| Repository file | Source path | Role |
| --- | --- | --- |
| `results/D4_integrated_report.md` | `D:\Codex\2026-06-28\c-0-20-c-seeds-20\outputs\D4_integrated_report.md` | D4a/D4b integrated summary |
| `results/D4b_residual_desync_results.md` | `D:\Codex\2026-06-28\c-0-20-c-seeds-20\outputs\D4b_residual_desync_results.md` | D4b grouped result summary |
| `results/d4a_repair_alignment.csv` | `D:\Codex\2026-06-28\c-0-20-c-seeds-20\outputs\protocol_d4_repair_null_20s200e\d4_repair_alignment.csv` | formal D4a alignment table |
| `source/d4b_residual_desync_gpu.py` | `D:\Codex\2026-06-28\c-0-20-c-seeds-20\outputs\d4b_residual_desync_gpu.py` | D4b implementation |
| `source/d4_complete_protocol.py` | `D:\Codex\2026-06-28\c-0-20-c-seeds-20\outputs\d4_complete_protocol.py` | unified protocol entry point |
| `source/D4_PROTOCOL.md` | `D:\Codex\2026-06-28\c-0-20-c-seeds-20\outputs\D4_PROTOCOL.md` | original commands and protocol notes |

## Missing formal artifact

The source archive contains `protocol_d4b_smoke/d4b_by_strength.csv`, but this is a smoke run and is not the formal 3000-configuration D4b output. It is therefore not used as formal evidence in `paper.md`.

## Reproducibility note

The source implementation currently calls `set_seed(...)` and constructs a new `SymbolicBottleneckModel` inside the loop over `C`. The manuscript treats this as part of the estimand definition and flags it as a release gate rather than silently interpreting the curve as a fixed-model intervention.
