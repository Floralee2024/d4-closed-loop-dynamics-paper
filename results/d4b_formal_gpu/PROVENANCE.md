# Formal D4b GPU release

This directory contains the formal D4b run used in the manuscript update. The raw table has one row per `(seed, residual_strength, variant, K, C, split)` and is merged from four non-overlapping seed shards.

- Raw rows: 3,000 = 10 seeds × 5 residual strengths × 1 variant × 2 K values × 10 C values × 3 splits.
- Seeds: `0..9`; residual strengths: `0, 0.1, 0.25, 0.5, 1.0`; `K ∈ {8, 16}`; `C ∈ {0, 0.5, 1, 2, 3, 5, 8, 12, 16, 24}`.
- Splits: `id`, `ood_random`, `ood_inverted`.
- Training protocol: 300 steps, batch size 512, CUDA AMP auto, TF32 enabled, DataLoader workers 0.
- Remote runtime: NVIDIA GeForce RTX 3080 Ti, PyTorch 2.8.0+cu128.
- Experiment source revision: `a9238aa` (`Add GPU sweep launcher and chunk merger`).
- `merge_manifest.json` and the four `chunk_*_config.json` files preserve the run arguments and shard provenance.
- `curves.csv` is the machine-readable source for all threshold summaries; `cstar_summary.csv` and the grouped CSVs are recomputed artifacts.

The run still uses the existing D4b estimand: each C condition trains a fresh model. These data close the raw-export gate, but they do not establish a fixed-trained-model intervention effect or the residual semantic mechanism. Nonzero residual conditions also have sparse or invalid `C*_util` thresholds under the preset gain gate, so utility sensitivity and threshold validity are reported separately.
