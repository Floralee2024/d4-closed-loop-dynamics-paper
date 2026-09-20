"""Merge seed-sharded D4b raw curves and recompute the official summaries."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_d4b_module(path: Path):
    spec = importlib.util.spec_from_file_location("d4b_residual_desync_gpu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load D4b module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("source/d4b_residual_desync_gpu.py"))
    parser.add_argument("--chunk-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_d4b_module(args.source)
    frames = []
    configs = []
    for chunk_dir in args.chunk_dirs:
        config_path = chunk_dir / "config.json"
        curves_path = chunk_dir / "curves.csv"
        if not config_path.exists() or not curves_path.exists():
            raise FileNotFoundError(f"Missing config.json or curves.csv in {chunk_dir}")
        with config_path.open("r", encoding="utf-8") as handle:
            configs.append(json.load(handle))
        frames.append(pd.read_csv(curves_path))
    first_args = configs[0]["args"]
    for config in configs[1:]:
        if config["args"] != first_args:
            raise ValueError("Chunk configurations differ; refusing to merge mismatched runs")
    curves = pd.concat(frames, ignore_index=True)
    key_cols = ["mode", "residual_lambda", "variant", "seed", "K", "C", "split"]
    duplicates = curves[curves.duplicated(key_cols, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Duplicate condition rows found: {len(duplicates)}")
    expected_per_chunk = (len(first_args["seeds"]) * len(first_args["residual_strengths"]) * len(first_args["variants"]) * len(first_args["Ks"]) * len(first_args["Cs"]) * len(first_args["splits"]))
    for frame, chunk_dir in zip(frames, args.chunk_dirs):
        if len(frame) != expected_per_chunk:
            raise ValueError(f"Incomplete chunk {chunk_dir}: {len(frame)} rows, expected {expected_per_chunk}")
    expected_total = expected_per_chunk * len(frames)
    if len(curves) != expected_total:
        raise ValueError(f"Merged row count {len(curves)} != expected {expected_total}")

    curves = module.add_dyn_scores(curves)
    summary = module.extract_cstars(
        curves,
        frac=first_args["frac"],
        min_gain=first_args["min_gain"],
        rank_min_gain=first_args["rank_min_gain"],
        dyn_min_gain=first_args["dyn_min_gain"],
    )
    aggregate_cols = [
        "E_rank", "E_dyn", "dyn_better", "C_rank", "C_dyn", "C_util",
        "C_rank_raw", "C_dyn_raw", "rank_gain", "dyn_gain", "util_gain",
        "rank_valid", "dyn_valid", "utility_sensitive", "utility_sensitivity",
        "Delta_C_dyn_eq", "sync_gap", "rank_util_gap", "dyn_util_gap", "dyn_after_eq",
    ]
    aggregate = summary.groupby(["mode", "residual_strength", "variant", "K", "split"])[aggregate_cols].agg(["mean", "std"]).reset_index()
    confirmation = module.confirmation_summary(summary, first_args["sync_tol"], first_args["early_c"], first_args["late_c"])
    confirmation_mode = module.mode_level_confirmation(summary, first_args["sync_tol"], first_args["early_c"], first_args["late_c"])

    by_strength = summary.copy()
    by_strength["negative_delta"] = (by_strength["Delta_C_dyn_eq"] < 0).astype(float)
    by_strength["positive_delta"] = (by_strength["Delta_C_dyn_eq"] > 0).astype(float)
    by_strength["sync"] = (by_strength["sync_gap"] <= first_args["sync_tol"]).astype(float)
    by_strength = by_strength.groupby(["residual_strength", "variant", "K", "split"]).agg(
        n=("sync_gap", "count"),
        C_rank_q90_mean=("C_rank", "mean"),
        C_dyn_q90_mean=("C_dyn", "mean"),
        C_util_q90_mean=("C_util", "mean"),
        delta_mean=("Delta_C_dyn_eq", "mean"),
        sync_gap_mean=("sync_gap", "mean"),
        sync_rate=("sync", "mean"),
        negative_delta_rate=("negative_delta", "mean"),
        positive_delta_rate=("positive_delta", "mean"),
        utility_sensitivity_mean=("utility_sensitivity", "mean"),
        utility_sensitive_rate=("utility_sensitive", "mean"),
        dyn_util_gap_mean=("dyn_util_gap", "mean"),
        rank_util_gap_mean=("rank_util_gap", "mean"),
        residual_var_ratio_mean=("residual_var_ratio", "mean"),
        residual_norm_ratio_mean=("residual_norm_ratio", "mean"),
    ).reset_index()
    strength_summary = by_strength.groupby(["residual_strength", "variant", "split"]).agg(
        n=("n", "sum"),
        sync_gap_mean=("sync_gap_mean", "mean"),
        sync_rate_mean=("sync_rate", "mean"),
        utility_sensitivity_mean=("utility_sensitivity_mean", "mean"),
        negative_delta_rate_mean=("negative_delta_rate", "mean"),
        positive_delta_rate_mean=("positive_delta_rate", "mean"),
        dyn_util_gap_mean=("dyn_util_gap_mean", "mean"),
        rank_util_gap_mean=("rank_util_gap_mean", "mean"),
    ).reset_index()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(args.out_dir / "curves.csv", index=False)
    summary.to_csv(args.out_dir / "cstar_summary.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate.csv", index=False)
    confirmation.to_csv(args.out_dir / "confirmation_summary.csv", index=False)
    confirmation_mode.to_csv(args.out_dir / "confirmation_by_mode.csv", index=False)
    by_strength.to_csv(args.out_dir / "d4b_by_strength.csv", index=False)
    strength_summary.to_csv(args.out_dir / "d4b_strength_summary.csv", index=False)
    merged_config = dict(configs[0])
    merged_config["args"] = dict(first_args)
    merged_config["args"]["seeds"] = sorted(int(x) for frame in frames for x in frame["seed"].unique())
    merged_config["merged_from"] = [str(path) for path in args.chunk_dirs]
    merged_config["merged_rows"] = int(len(curves))
    with (args.out_dir / "merge_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(merged_config, handle, indent=2, ensure_ascii=False)
    print(f"merged {len(curves)} raw rows from {len(frames)} chunks into {args.out_dir}")


if __name__ == "__main__":
    main()
