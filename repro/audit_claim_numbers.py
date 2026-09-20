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


def audit_d4a():
    rows = read_csv("d4a_repair_alignment.csv")
    assert len(rows) == 8, len(rows)
    close(mean(float(row["dyn_to_util_q90_err_mean"]) for row in rows), 0.3125)
    close(mean(float(row["dyn_null_to_util_q90_err_mean"]) for row in rows), 3.2125)
    close(mean(float(row["rank_to_util_q90_err_mean"]) for row in rows), 12.76875)
    close(mean(float(row["dyn_closer_q90_rate"]) for row in rows), 0.9375)
    return {"rows": len(rows), "dyn_closer_q90_rate": 0.9375}


def audit_d4b():
    rows = read_csv("d4b_strength_summary_report.csv")
    assert len(rows) == 12, len(rows)
    strengths = sorted({float(row["residual_strength"]) for row in rows})
    splits = sorted({row["split"] for row in rows})
    assert strengths == [0.0, 0.1, 0.5, 1.0], strengths
    assert splits == ["id", "ood_inverted", "ood_random"], splits
    zero = [float(row["sync_gap"]) for row in rows if float(row["residual_strength"]) == 0.0]
    nonzero = [float(row["sync_gap"]) for row in rows if float(row["residual_strength"]) > 0.0]
    assert min(nonzero) > max(zero), (min(nonzero), max(zero))
    return {"rows": len(rows), "zero_sync_gap_range": [min(zero), max(zero)], "nonzero_sync_gap_range": [min(nonzero), max(nonzero)]}


if __name__ == "__main__":
    print({"d4a": audit_d4a(), "d4b_report_summary": audit_d4b()})
    print("Claim-number audit passed.")
