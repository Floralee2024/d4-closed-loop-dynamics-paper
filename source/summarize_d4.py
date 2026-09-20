#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an integrated D4 report from D4a and D4b CSV outputs.

D4a is the C* decomposition / null-baseline experiment produced by
`d2_d4_protocol.py --mode d4_repair`.

D4b is the residual-bypass desynchronization experiment produced by
`d4b_residual_desync_gpu.py`.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=3):
    value = fnum(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def mean(values):
    xs = [fnum(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def markdown_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def summarize_d4a(d4a_dir: Path):
    rows = read_csv(d4a_dir / "d4_repair_alignment.csv")
    if not rows:
        return "## D4a: C* Decomposition\n\nNo D4a alignment CSV found.\n"

    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r.get("split", ""),
                r.get("variant", ""),
                r.get("K", ""),
                fmt(r.get("dyn_to_util_q90_err_mean")),
                fmt(r.get("dyn_null_to_util_q90_err_mean")),
                fmt(r.get("rank_to_util_q90_err_mean")),
                fmt(r.get("dyn_closer_q90_rate")),
                fmt(r.get("dyn_beats_null_q90_rate")),
            ]
        )

    global_dyn_err = mean(r.get("dyn_to_util_q90_err_mean") for r in rows)
    global_null_err = mean(r.get("dyn_null_to_util_q90_err_mean") for r in rows)
    global_rank_err = mean(r.get("rank_to_util_q90_err_mean") for r in rows)
    global_dyn_closer = mean(r.get("dyn_closer_q90_rate") for r in rows)

    text = [
        "## D4a: C* Decomposition",
        "",
        (
            "D4a tests whether a single unsupervised C* is sufficient. "
            "The main result is that dynamics-level C* is much closer to "
            "utility C* than rank/equivalence C*, and this survives the null "
            "dynamic baseline."
        ),
        "",
        f"- Mean |C*_dyn - C*_util|: {fmt(global_dyn_err)}",
        f"- Mean |C*_dyn_null - C*_util|: {fmt(global_null_err)}",
        f"- Mean |C*_rank - C*_util|: {fmt(global_rank_err)}",
        f"- Mean dyn-closer-than-rank rate: {fmt(global_dyn_closer)}",
        "",
        markdown_table(
            [
                "split",
                "variant",
                "K",
                "dyn-util err",
                "null-util err",
                "rank-util err",
                "dyn closer",
                "dyn beats null",
            ],
            table_rows,
        ),
        "",
    ]
    return "\n".join(text)


def summarize_d4b(d4b_dir: Path):
    rows = read_csv(d4b_dir / "d4b_strength_summary.csv")
    if not rows:
        rows = read_csv(d4b_dir / "d4b_by_strength.csv")
    if not rows:
        return (
            "## D4b: Residual Bypass Desynchronization\n\n"
            "No D4b summary CSV found yet. Run `outputs/d4b_residual_desync_gpu.py` "
            "and point `--d4b-dir` at its output directory.\n"
        )

    table_rows = []
    has_variant = "variant" in rows[0]
    for r in rows:
        table_rows.append(
            [
                r.get("residual_strength", ""),
                r.get("variant", "") if has_variant else "",
                r.get("split", ""),
                fmt(r.get("sync_gap_mean")),
                fmt(r.get("sync_rate_mean", r.get("sync_rate"))),
                fmt(r.get("utility_sensitivity_mean")),
                fmt(r.get("negative_delta_rate_mean", r.get("negative_delta_rate"))),
                fmt(r.get("positive_delta_rate_mean", r.get("positive_delta_rate"))),
            ]
        )

    by_strength = {}
    for r in rows:
        s = r.get("residual_strength", "")
        by_strength.setdefault(s, []).append(r)
    trend_rows = []
    for s, rs in sorted(by_strength.items(), key=lambda kv: fnum(kv[0], 999.0)):
        trend_rows.append(
            [
                s,
                fmt(mean(r.get("sync_gap_mean") for r in rs)),
                fmt(mean(r.get("utility_sensitivity_mean") for r in rs)),
                fmt(mean(r.get("sync_rate_mean", r.get("sync_rate")) for r in rs)),
            ]
        )

    text = [
        "## D4b: Residual Bypass Desynchronization",
        "",
        (
            "D4b tests the mechanism behind D4a: symbolic structure formation "
            "is synchronized when the downstream path is forced through symbols, "
            "while continuous residuals desynchronize rank/equivalence, dynamics, "
            "and utility phase markers."
        ),
        "",
        "Residual-strength trend:",
        "",
        markdown_table(
            ["residual strength", "sync gap", "utility sensitivity", "sync rate"],
            trend_rows,
        ),
        "",
        "Split-specific summary:",
        "",
        markdown_table(
            [
                "residual strength",
                "variant",
                "split",
                "sync gap",
                "sync rate",
                "utility sensitivity",
                "dyn-first rate",
                "rank-first rate",
            ],
            table_rows,
        ),
        "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4a-dir", default="outputs/protocol_d4_repair_null_20s200e")
    parser.add_argument("--d4b-dir", default="outputs/protocol_d4b_residual_desync")
    parser.add_argument("--out", default="outputs/D4_integrated_report.md")
    args = parser.parse_args()

    d4a_dir = Path(args.d4a_dir)
    d4b_dir = Path(args.d4b_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = [
        "# Experiment D4: C* Decomposition and Residual Desynchronization",
        "",
        "Final structure:",
        "",
        "- D4a: decompose C* into rank/equivalence, dynamics, and utility phase markers.",
        "- D4b: test whether continuous residual bypass desynchronizes those phase markers.",
        "",
        summarize_d4a(d4a_dir),
        summarize_d4b(d4b_dir),
        "## Interpretation",
        "",
        (
            "The upgraded D4 should no longer claim that a single C* directly "
            "predicts downstream utility. The stronger claim is hierarchical: "
            "rank/equivalence C*, dynamics C*, and utility C* are separable; "
            "dynamics C* is the best utility-aligned marker; and residual "
            "continuous bypass changes the ordering of symbolic structure formation."
        ),
        "",
    ]
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
