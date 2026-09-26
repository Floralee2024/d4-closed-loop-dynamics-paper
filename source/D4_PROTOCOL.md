> **Historical protocol record.** This file preserves the original unified D4 naming and hypotheses for reproducibility. The current manuscript treats D4a and D4b as related but non-identical studies and does not adopt mechanism or causal claims from this protocol text. See paper.md and CLAIMS_AND_LIMITS.md for the current evidence boundary.
# Experiment D4 Protocol

## Complete Entry Point

Use this script as the unified D4 code version:

```bash
python outputs/d4_complete_protocol.py --stage smoke --device auto
```

Full run:

```bash
python outputs/d4_complete_protocol.py --stage all \
  --device cuda \
  --tf32 \
  --persistent-workers \
  --d4a-dir outputs/protocol_d4_repair_null_20s200e \
  --d4b-dir outputs/protocol_d4b_residual_desync \
  --report outputs/D4_integrated_report.md
```

Run only one stage:

```bash
python outputs/d4_complete_protocol.py --stage d4a --device auto
python outputs/d4_complete_protocol.py --stage d4b --device cuda --tf32 --persistent-workers
python outputs/d4_complete_protocol.py --stage report
```

The complete entry point delegates to the underlying scripts below, keeping D4a
and D4b reproducible as separate experiment modules while presenting one
protocol-level command.

## D4a: C* Decomposition

Question:

Does one unsupervised C* explain downstream utility?

Answer used by the current protocol:

No. D4 is decomposed into three phase markers:

- `C*_rank`: representation geometry / symbolic equivalence.
- `C*_dyn`: dynamics-level symbolic structure.
- `C*_util`: downstream utility phase point.

Main readout:

- `dyn_to_util_q90_err`
- `dyn_null_to_util_q90_err`
- `rank_to_util_q90_err`
- `dyn_closer_q90_rate`
- `dyn_beats_null_q90_rate`

Canonical run:

```bash
python outputs/d2_d4_protocol.py --mode d4_repair \
  --outdir outputs/protocol_d4_repair_null_20s200e \
  --splits id ood_inverted \
  --variants original pre_softmax \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --Ks 8 16 \
  --C-grid 0 0.5 1 2 3 5 8 12 16 24 \
  --epochs 200 \
  --min-utility-gain 0.10 \
  --min-structure-gain 0.02 \
  --structure-n 384 \
  --device auto
```

## D4b: Residual Bypass Desynchronization

Question:

Why do `C*_rank`, `C*_dyn`, and `C*_util` separate?

Hypothesis:

When the downstream path is forced through symbols, symbolic equivalence and
dynamics structure synchronize. Continuous residual bypass progressively breaks
that synchronization and lowers utility sensitivity to C.

Main readout:

- `sync_gap = |C*_dyn - C*_rank|`
- `Delta_C_dyn_eq = C*_dyn - C*_rank`
- `utility_sensitivity`
- `sync_rate`
- `negative_delta_rate`
- `positive_delta_rate`
- split-specific ID/OOD ordering

Canonical GPU run:

```bash
python outputs/d4b_residual_desync_gpu.py \
  --steps 300 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --Ks 8 16 \
  --Cs 0 0.5 1 2 3 5 8 12 16 24 \
  --variants original \
  --residual-strengths 0 0.1 0.25 0.5 1.0 \
  --splits id ood_random ood_inverted \
  --device cuda \
  --amp auto \
  --tf32 \
  --batch-size 512 \
  --eval-batch-size 2048 \
  --num-workers 2 \
  --persistent-workers \
  --out-dir outputs/protocol_d4b_residual_desync
```

Smoke test:

```bash
python outputs/d4b_residual_desync_gpu.py \
  --steps 30 \
  --seeds 0 \
  --Ks 8 \
  --Cs 0 1 3 8 \
  --residual-strengths 0 0.25 1.0 \
  --splits ood_inverted \
  --out-dir outputs/protocol_d4b_smoke \
  --device auto \
  --amp off \
  --no-plots
```

## Integrated Report

After D4a and D4b have both been run:

```bash
python outputs/summarize_d4.py \
  --d4a-dir outputs/protocol_d4_repair_null_20s200e \
  --d4b-dir outputs/protocol_d4b_residual_desync \
  --out outputs/D4_integrated_report.md
```

Final D4 claim:

The single-C* view is insufficient. Rank/equivalence, dynamics structure, and
downstream utility are separable phase markers. Dynamics C* is the best
utility-aligned marker under D4a. D4b explains the separation mechanism:
continuous residual bypass changes the ordering of symbolic structure formation
rather than simply helping or hurting performance.
