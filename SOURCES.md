# Source provenance

The files in this repository are copied from the local D4 experiment archive. Source and result artifacts are kept distinct from the revised manuscript. Historical markdown and protocol copies may carry a status header added during revision; underlying numerical CSV outputs and experiment logic are not rewritten.

## Local source paths

| Repository file | Source path | Role |
| --- | --- | --- |
| `results/D4_integrated_report.md` | `local D4 archive/outputs/D4_integrated_report.md` | D4a/D4b integrated summary |
| `results/D4b_residual_desync_results.md` | `local D4 archive/outputs/D4b_residual_desync_results.md` | D4b grouped result summary |
| `results/d4a_repair_alignment.csv` | `local D4 archive/outputs/protocol_d4_repair_null_20s200e/d4_repair_alignment.csv` | formal D4a alignment table |
| `source/d4b_residual_desync_gpu.py` | `local D4 archive/outputs/d4b_residual_desync_gpu.py` | D4b implementation |
| `source/d4_complete_protocol.py` | `local D4 archive/outputs/d4_complete_protocol.py` | unified protocol entry point |
| `source/d2_d4_protocol.py` | `local D4 archive/outputs/d2_d4_protocol.py` | D4a runner |
| `source/experiment_D_pytorch_history_symbol_budget.py` | `local D4 archive/outputs/experiment_D_pytorch_history_symbol_budget.py` | D4a model dependency |
| `source/summarize_d4.py` | `local D4 archive/outputs/summarize_d4.py` | integrated report generator |
| `source/D4_PROTOCOL.md` | `local D4 archive/outputs/D4_PROTOCOL.md` | original commands and protocol notes |
| `repro/fixed_model_sweep.py` | repository-local diagnostic | fixed-model/common-random-number probe |
| `results/d4b_formal_gpu/` | AutoDL RTX 3080 Ti run, merged locally from four shards | formal 3,000-row D4b raw table and summaries |

## Repository-local reproducibility patch

The repository copy of experiment_D_pytorch_history_symbol_budget.py changes
`ridge_mse` from a direct solve of `X.T @ X` to an equivalent augmented
least-squares solve. The original CPU smoke failed on a singular Gram matrix;
the augmented formulation preserves the unregularized intercept and ridge
penalty while remaining stable for rank-deficient symbolic features. Archived
formal D4a numbers were not silently recomputed after this patch.

## Formal D4b artifact

The formal GPU release contains one raw row per `(seed, residual_strength, variant, K, C, split)`, four shard configs, a merge manifest, and recomputed summaries. A local independent merge reproduced the remote summaries exactly: 3,000 raw rows and 300 seed-level threshold rows. The source archive also contains `protocol_d4b_smoke/d4b_by_strength.csv`; that smoke run is kept separate and was not merged into the formal table.

## Reproducibility note

The D4b source implementation currently calls `set_seed(...)` and constructs a new `SymbolicBottleneckModel` inside the loop over `C`. The manuscript treats this as part of the estimand definition and keeps the fixed-model/common-random-number analysis as an open causal gate rather than silently interpreting the curve as a fixed-model intervention.

## Revision recovery and post-hoc analysis (26 September 2026)

`results/d4a_formal_archive/` copies the four CSVs from the recorded `protocol_d4_repair_null_20s200e` archive directory above without changing their bytes. The alignment SHA-256 matches the existing release. Its provenance manifest records each file hash; `verification.json` records independent q90 extraction checks. No smoke or other-run files were substituted. Historical report and protocol copies carry explicit status headers; their numerical bodies remain archival.

`results/revision_posthoc/` contains exploratory analyses of existing D4b data, with input and script hashes. Original experiment outputs and training code are unchanged. Current methods and claim boundaries supersede archived reports, including their earlier mechanism wording.
