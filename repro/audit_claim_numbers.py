"""Recompute the headline numbers used by paper.md from repository artifacts."""

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def close(actual, expected, tol=1e-9):
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"expected {expected}, got {actual}")


def read_csv(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_result_csv(name):
    with (ROOT / "results" / "d4b_formal_gpu" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_d4a():
    rows = read_csv("d4a_repair_alignment.csv")
    assert len(rows) == 8, len(rows)
    close(mean(float(row["dyn_to_util_q90_err_mean"]) for row in rows), 0.3125)
    close(mean(float(row["dyn_null_to_util_q90_err_mean"]) for row in rows), 3.2125)
    close(mean(float(row["rank_to_util_q90_err_mean"]) for row in rows), 12.76875)
    close(mean(float(row["dyn_closer_q90_rate"]) for row in rows), 0.9375)
    return {"rows": len(rows), "dyn_closer_q90_rate": 0.9375}


def audit_d4b():
    raw = read_result_csv("curves.csv")
    summary = read_result_csv("cstar_summary.csv")
    rows = read_result_csv("d4b_strength_summary.csv")
    assert len(raw) == 3000, len(raw)
    keys = ["mode", "residual_lambda", "variant", "seed", "K", "C", "split"]
    assert len({tuple(row[key] for key in keys) for row in raw}) == 3000
    assert len(summary) == 300, len(summary)
    assert len(rows) == 15, len(rows)
    strengths = sorted({float(row["residual_strength"]) for row in rows})
    splits = sorted({row["split"] for row in rows})
    assert strengths == [0.0, 0.1, 0.25, 0.5, 1.0], strengths
    assert splits == ["id", "ood_inverted", "ood_random"], splits
    zero = [float(row["sync_gap_mean"]) for row in rows if float(row["residual_strength"]) == 0.0]
    nonzero = [float(row["sync_gap_mean"]) for row in rows if float(row["residual_strength"]) > 0.0]
    assert min(nonzero) > max(zero), (min(nonzero), max(zero))
    zero_sensitivity = [float(row["utility_sensitivity_mean"]) for row in rows if float(row["residual_strength"]) == 0.0]
    nonzero_sensitivity = [float(row["utility_sensitivity_mean"]) for row in rows if float(row["residual_strength"]) > 0.0]
    assert max(nonzero_sensitivity) < min(zero_sensitivity), (max(nonzero_sensitivity), min(zero_sensitivity))
    return {"raw_rows": len(raw), "summary_rows": len(summary), "group_rows": len(rows), "zero_sync_gap_range": [min(zero), max(zero)], "nonzero_sync_gap_range": [min(nonzero), max(nonzero)], "zero_utility_sensitivity_range": [min(zero_sensitivity), max(zero_sensitivity)], "nonzero_utility_sensitivity_range": [min(nonzero_sensitivity), max(nonzero_sensitivity)]}


if __name__ == "__main__":
    print({"d4a": audit_d4a(), "d4b_formal": audit_d4b()})
    print("Claim-number audit passed.")
