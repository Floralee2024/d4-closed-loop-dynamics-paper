"""Small fixed-model/common-random-number sweep for the D4b claim gate.

This script trains one model per (seed, residual strength, K, variant) at a
pre-specified ``--train-C`` and then evaluates the same weights over the C
grid. Evaluation datasets are constructed once per seed and reused across C.
It is deliberately separate from the archived D4b runner, whose estimand is
per-C retraining.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "d4b_residual_desync_gpu.py"


def load_d4b():
    spec = importlib.util.spec_from_file_location("d4b_residual_desync_gpu", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/fixed_model_sweep")
    p.add_argument("--device", default="cpu")
    p.add_argument("--amp", choices=["off", "auto", "fp16", "bf16"], default="off")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--Ks", nargs="+", type=int, default=[8])
    p.add_argument("--Cs", nargs="+", type=float, default=[0, 1, 3, 8])
    p.add_argument("--residual-strengths", nargs="+", type=float, default=[0, 0.5, 1.0])
    p.add_argument("--splits", nargs="+", default=["id", "ood_inverted"])
    p.add_argument("--variant", default="original", choices=["original", "pre_softmax"])
    p.add_argument("--train-C", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--c-scale", type=float, default=2.0)
    p.add_argument("--world-seed", type=int, default=1234)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--min-gain", type=float, default=0.10)
    p.add_argument("--rank-min-gain", type=float, default=0.05)
    p.add_argument("--dyn-min-gain", type=float, default=0.05)
    p.add_argument("--frac", type=float, default=0.90)
    p.add_argument(
        "--dynamics-metric",
        choices=["hard_id_nmi", "transition_head_gain"],
        default="transition_head_gain",
        help="C-dependent transition-head gain is the valid fixed-model diagnostic; hard_id_nmi reproduces the archived metric.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    mod = load_d4b()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.torch_threads > 0:
        mod.torch.set_num_threads(args.torch_threads)
    device = mod.choose_device(args.device)
    mod.configure_gpu(device)
    amp_enabled, amp_dtype, scaler_enabled = mod.resolve_amp(device, args.amp)
    world = mod.WorldSpec(obs_dim=args.obs_dim, seed=args.world_seed).build()
    train_cfg = mod.TrainConfig(steps=args.steps, batch_size=args.batch_size)
    rows = []

    for seed in args.seeds:
        train_ds = mod.StagedDataset(args.n_train, args.seq_len, world, seed=10000 + seed, split="train_mixed")
        eval_sets = {
            split: mod.StagedDataset(args.n_test, args.seq_len, world, seed=20000 + 17 * seed + i, split=split)
            for i, split in enumerate(args.splits)
        }
        for residual_strength in args.residual_strengths:
            for K in args.Ks:
                # Same initialization seed across residual strengths; the
                # residual path is the intended condition difference.
                mod.set_seed(seed + 1000 * K + (0 if args.variant == "original" else 7))
                model = mod.SymbolicBottleneckModel(
                    obs_dim=args.obs_dim,
                    K=K,
                    n_actions=world.n_actions,
                    n_outcomes=world.n_outcomes,
                    variant=args.variant,
                    hidden_dim=args.hidden_dim,
                    embed_dim=args.embed_dim,
                    c_scale=args.c_scale,
                    mode="residual_bypass",
                    residual_lambda=residual_strength,
                )
                print(f"train seed={seed} residual={residual_strength:g} K={K} at C={args.train_C:g}", flush=True)
                mod.train_one(
                    model,
                    train_ds,
                    C=args.train_C,
                    cfg=train_cfg,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    scaler_enabled=scaler_enabled,
                    num_workers=0,
                )
                for C in args.Cs:
                    for split, ds in eval_sets.items():
                        metrics = mod.evaluate(
                            model,
                            ds,
                            C=C,
                            device=device,
                            batch_size=args.eval_batch_size,
                            amp_enabled=amp_enabled,
                            amp_dtype=amp_dtype,
                            num_workers=0,
                        )
                        metrics.update(
                            {
                                "seed": seed,
                                "mode": "fixed_model",
                                "residual_lambda": residual_strength,
                                "residual_strength": residual_strength,
                                "variant": args.variant,
                                "K": K,
                                "C": C,
                                "train_C": args.train_C,
                                "split": split,
                            }
                        )
                        rows.append(metrics)

    curves = pd.DataFrame(rows)
    curves = mod.add_dyn_scores(curves)
    if args.dynamics_metric == "transition_head_gain":
        group_cols = ["mode", "residual_lambda", "variant", "seed", "K", "split"]
        for _, group in curves.groupby(group_cols, sort=False):
            idx = group.index
            curves.loc[idx, "dyn_score"] = mod.normalize_curve_values(group["transition_head_gain"].to_numpy())
    summary = mod.extract_cstars(
        curves,
        frac=args.frac,
        min_gain=args.min_gain,
        rank_min_gain=args.rank_min_gain,
        dyn_min_gain=args.dyn_min_gain,
    )
    curves.to_csv(output / "curves.csv", index=False)
    summary.to_csv(output / "cstar_summary.csv", index=False)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(f"Saved fixed-model curves and C* summary to {output}")
    print(summary[["seed", "residual_strength", "K", "split", "C_rank", "C_dyn", "C_util", "sync_gap", "utility_sensitivity"]].to_string(index=False))


if __name__ == "__main__":
    main()
