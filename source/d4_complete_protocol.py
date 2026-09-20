#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified D4 protocol runner.

This file is the complete D4 entry point. It combines:

- D4a: C* decomposition + null dynamic baseline
  implemented in `d2_d4_protocol.py --mode d4_repair`.

- D4b: residual bypass desynchronization
  implemented in `d4b_residual_desync_gpu.py`.

- Integrated reporting
  implemented in `summarize_d4.py`.

The underlying experiment files remain separate so D4a can reuse the existing
history/state model and D4b can remain GPU-friendly and self-contained.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
D4A_SCRIPT = HERE / "d2_d4_protocol.py"
D4B_SCRIPT = HERE / "d4b_residual_desync_gpu.py"
REPORT_SCRIPT = HERE / "summarize_d4.py"


DEFAULT_C_GRID = ["0", "0.5", "1", "2", "3", "5", "8", "12", "16", "24"]
DEFAULT_D4A_SEEDS = [str(i) for i in range(20)]
DEFAULT_D4B_SEEDS = [str(i) for i in range(10)]


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(cmd, *, dry_run=False):
    print("\n$ " + quote_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def split_words(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []
    return [x for x in str(value).replace(",", " ").split() if x]


def build_d4a_cmd(args):
    outdir = Path(args.d4a_dir)
    seeds = split_words(args.d4a_seeds)
    c_grid = split_words(args.C_grid)
    variants = split_words(args.d4a_variants)
    splits = split_words(args.d4a_splits)
    ks = split_words(args.Ks)

    return [
        sys.executable,
        D4A_SCRIPT,
        "--mode",
        "d4_repair",
        "--outdir",
        outdir,
        "--splits",
        *splits,
        "--variants",
        *variants,
        "--seeds",
        *seeds,
        "--Ks",
        *ks,
        "--C-grid",
        *c_grid,
        "--epochs",
        str(args.d4a_epochs),
        "--min-utility-gain",
        str(args.min_utility_gain),
        "--min-structure-gain",
        str(args.min_structure_gain),
        "--structure-n",
        str(args.structure_n),
        "--device",
        args.device,
    ]


def build_d4b_cmd(args):
    outdir = Path(args.d4b_dir)
    seeds = split_words(args.d4b_seeds)
    c_grid = split_words(args.C_grid)
    variants = split_words(args.d4b_variants)
    splits = split_words(args.d4b_splits)
    ks = split_words(args.Ks)
    residual_strengths = split_words(args.residual_strengths)

    cmd = [
        sys.executable,
        D4B_SCRIPT,
        "--out-dir",
        outdir,
        "--device",
        args.device,
        "--amp",
        args.amp,
        "--seeds",
        *seeds,
        "--Ks",
        *ks,
        "--Cs",
        *c_grid,
        "--variants",
        *variants,
        "--residual-strengths",
        *residual_strengths,
        "--splits",
        *splits,
        "--steps",
        str(args.d4b_steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--min-gain",
        str(args.min_utility_gain),
        "--rank-min-gain",
        str(args.rank_min_gain),
        "--dyn-min-gain",
        str(args.dyn_min_gain),
        "--sync-tol",
        str(args.sync_tol),
    ]
    if args.tf32:
        cmd.append("--tf32")
    if args.persistent_workers:
        cmd.append("--persistent-workers")
    if args.no_plots:
        cmd.append("--no-plots")
    if args.include_continuous_only:
        cmd.append("--include-continuous-only")
    return cmd


def build_report_cmd(args):
    return [
        sys.executable,
        REPORT_SCRIPT,
        "--d4a-dir",
        Path(args.d4a_dir),
        "--d4b-dir",
        Path(args.d4b_dir),
        "--out",
        Path(args.report),
    ]


def apply_smoke_defaults(args):
    args.d4a_dir = str(Path(args.out_root) / "protocol_d4a_smoke")
    args.d4b_dir = str(Path(args.out_root) / "protocol_d4b_smoke")
    args.report = str(Path(args.out_root) / "D4_smoke_report.md")
    args.d4a_seeds = "0"
    args.d4b_seeds = "0"
    args.Ks = "8"
    args.C_grid = "0 1 3 8"
    args.d4a_epochs = 5
    args.d4b_steps = 30
    args.d4a_splits = "id"
    args.d4b_splits = "ood_inverted"
    args.d4a_variants = "original"
    args.d4b_variants = "original"
    args.residual_strengths = "0 0.25 1.0"
    args.amp = "off"
    args.no_plots = True
    args.num_workers = 0
    args.persistent_workers = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Complete D4 runner: D4a decomposition, D4b residual desync, and integrated report."
    )
    p.add_argument("--stage", choices=["d4a", "d4b", "report", "all", "smoke"], default="report")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--d4a-dir", default="outputs/protocol_d4_repair_null_20s200e")
    p.add_argument("--d4b-dir", default="outputs/protocol_d4b_residual_desync")
    p.add_argument("--report", default="outputs/D4_integrated_report.md")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--Ks", default="8 16")
    p.add_argument("--C-grid", default=" ".join(DEFAULT_C_GRID))
    p.add_argument("--min-utility-gain", type=float, default=0.10)

    p.add_argument("--d4a-seeds", default=" ".join(DEFAULT_D4A_SEEDS))
    p.add_argument("--d4a-splits", default="id ood_inverted")
    p.add_argument("--d4a-variants", default="original pre_softmax")
    p.add_argument("--d4a-epochs", type=int, default=200)
    p.add_argument("--min-structure-gain", type=float, default=0.02)
    p.add_argument("--structure-n", type=int, default=384)

    p.add_argument("--d4b-seeds", default=" ".join(DEFAULT_D4B_SEEDS))
    p.add_argument("--d4b-splits", default="id ood_random ood_inverted")
    p.add_argument("--d4b-variants", default="original")
    p.add_argument("--d4b-steps", type=int, default=300)
    p.add_argument("--residual-strengths", default="0 0.1 0.25 0.5 1.0")
    p.add_argument("--amp", choices=["off", "auto", "fp16", "bf16"], default="auto")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--include-continuous-only", action="store_true")
    p.add_argument("--rank-min-gain", type=float, default=0.05)
    p.add_argument("--dyn-min-gain", type=float, default=0.05)
    p.add_argument("--sync-tol", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "smoke":
        apply_smoke_defaults(args)
        stages = ["d4a", "d4b", "report"]
    elif args.stage == "all":
        stages = ["d4a", "d4b", "report"]
    else:
        stages = [args.stage]

    for stage in stages:
        if stage == "d4a":
            run_cmd(build_d4a_cmd(args), dry_run=args.dry_run)
        elif stage == "d4b":
            run_cmd(build_d4b_cmd(args), dry_run=args.dry_run)
        elif stage == "report":
            run_cmd(build_report_cmd(args), dry_run=args.dry_run)
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    main()
