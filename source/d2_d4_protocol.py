"""
D2/D4 protocol runner with an adapter over the existing PyTorch experiment.

Recommended order:
  1. Gate ID/OOD signal validity:
     python d2_d4_protocol.py --mode gate --outdir outputs/protocol_gate \
       --splits id ood --variants original pre_softmax --seeds 0 1 2 3 4 5 6 7 8 9

  2. D2 on ID:
     python d2_d4_protocol.py --mode d2 --outdir outputs/protocol_d2_id \
       --splits id --variants original pre_softmax --seeds 0 1 ... 19 \
       --Ks 4 8 16 64 --epochs 200

  3. D4 dense C/C* curve on ID:
     python d2_d4_protocol.py --mode d4 --outdir outputs/protocol_d4_id \
       --splits id --variants original pre_softmax --seeds 0 1 ... 9 \
       --Ks 8 16 --rel-Cs 0 0.25 0.5 0.75 1 1.25 1.5 2 --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics as stats
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXP_PATH = HERE / "experiment_D_pytorch_history_symbol_budget.py"
spec = importlib.util.spec_from_file_location("history_budget_exp", EXP_PATH)
exp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp)


BOTTLENECKS = ["discrete", "cont_log2", "cont_K", "random_symbol"]


@dataclass
class ProtocolConfig:
    epochs: int
    min_full_gain: float


class MyAdapter:
    def __init__(self, cfg: ProtocolConfig):
        self.cfg = cfg
        self._models = {}
        self._cstars = {}
        self._eval_cache = {}
        self._structure_cache = {}

    @staticmethod
    def split_to_dataset(split: str) -> str:
        mapping = {
            "id": "id",
            "train": "id",
            "ood": "ood_inverted",
            "ood_inverted": "ood_inverted",
            "ood_mild": "ood_mild",
            "ood_random": "ood_random",
        }
        if split not in mapping:
            raise ValueError(f"Unknown split: {split}")
        return mapping[split]

    def model(self, *, variant: str, seed: int):
        key = (variant, seed)
        if key not in self._models:
            print(f"[train] variant={variant} seed={seed} epochs={self.cfg.epochs}", flush=True)
            self._models[key] = exp.train_model(variant, seed=seed, epochs=self.cfg.epochs)
        return self._models[key]

    def estimate_c_star(self, *, variant: str, seed: int, K=None, split="id") -> float:
        key = (variant, seed, K, split)
        if key not in self._cstars:
            model = self.model(variant=variant, seed=seed)
            k_unsup = 16 if K is None else int(K)
            info = exp.find_unsup_c_star(model, seed=50000 + seed + 17 * k_unsup, k=k_unsup)
            c_star = max(float(info["C_star"]), 0.5)
            self._cstars[key] = {
                "C_star": c_star,
                "usage_entropy": float(info["usage_entropy"]),
                "rank_norm": float(info["rank_norm"]),
                "score": float(info["score"]),
            }
            print(
                f"[cstar] variant={variant} seed={seed} K={K} split={split} "
                f"C*={c_star:.3f} ent={info['usage_entropy']:.3f} rank={info['rank_norm']:.3f}",
                flush=True,
            )
        return self._cstars[key]["C_star"]

    def cstar_info(self, *, variant: str, seed: int, K=None, split="id"):
        self.estimate_c_star(variant=variant, seed=seed, K=K, split=split)
        return self._cstars[(variant, seed, K, split)]

    def eval_all_budgets(self, *, split: str, variant: str, seed: int, C: float):
        key = (split, variant, seed, round(float(C), 8))
        if key in self._eval_cache:
            return self._eval_cache[key]

        model = self.model(variant=variant, seed=seed)
        z_train, y_train = exp.flatten_states(model, C, seed + 1, split="train_mixed", n=exp.EVAL_TRAIN_N)
        test_split = self.split_to_dataset(split)
        z_test, y_test = exp.flatten_states(model, C, seed + 2, split=test_split, n=exp.EVAL_TEST_N)
        rows_by_k = exp.eval_budget_for_split(z_train, y_train, z_test, y_test, seed + 10000, split.upper())
        self._eval_cache[key] = rows_by_k
        return rows_by_k

    def eval_baselines(self, *, split: str, seed: int, variant="original", C=None):
        if C is None:
            C = self.estimate_c_star(variant=variant, seed=seed, split=split)
        rows_by_k = self.eval_all_budgets(split=split, variant=variant, seed=seed, C=C)
        row = rows_by_k[max(rows_by_k.keys())]
        tag = split.upper()
        return {
            "mse_random": float(row[f"mse_random_{tag}"]),
            "mse_majority": float(row[f"mse_majority_{tag}"]),
            "mse_full": float(row[f"mse_full_{tag}"]),
        }

    def train_and_eval(
        self,
        *,
        split: str,
        variant: str,
        seed: int,
        K: int,
        bottleneck: str,
        C: float,
        epochs: int,
        extra=None,
    ):
        rows_by_k = self.eval_all_budgets(split=split, variant=variant, seed=seed, C=C)
        row = rows_by_k[int(K)]
        tag = split.upper()

        if bottleneck == "discrete":
            mse_model = row[f"mse_discrete_{tag}"]
        elif bottleneck == "cont_log2":
            mse_model = row[f"mse_continuous_log2_{tag}"]
        elif bottleneck == "cont_K":
            mse_model = row[f"mse_continuous_K_{tag}"]
        elif bottleneck == "random_symbol":
            mse_model = None
            # Convert random-symbol utility back to an MSE for uniform reporting.
            u = row[f"U_random_symbol_{tag}"]
            mse_random = row[f"mse_random_{tag}"]
            mse_full = row[f"mse_full_{tag}"]
            mse_model = mse_random - u * max(mse_random - mse_full, 1e-6)
        else:
            raise ValueError(f"Unknown bottleneck: {bottleneck}")

        return {
            "mse_model": float(mse_model),
            "mse_random": float(row[f"mse_random_{tag}"]),
            "mse_full": float(row[f"mse_full_{tag}"]),
            "low_signal": int(row[f"low_signal_{tag}"]),
            "U": float(row[f"U_{'discrete' if bottleneck == 'discrete' else 'continuous_log2' if bottleneck == 'cont_log2' else 'continuous_K' if bottleneck == 'cont_K' else 'random_symbol'}_{tag}"]),
        }


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def sem(xs):
    if len(xs) <= 1:
        return 0.0
    return float(np.std(xs, ddof=1) / math.sqrt(len(xs)))


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def full_gain_ratio(mse_random, mse_full):
    if mse_random <= 1e-12:
        return 0.0
    return max(0.0, (mse_random - mse_full) / mse_random)


def gated_baseline(mse_random, mse_majority):
    if mse_majority < mse_random:
        return mse_majority, "majority"
    return mse_random, "mean"


def entropy_from_probs(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def normalized_mi(labels_a, labels_b):
    a = np.asarray(labels_a, dtype=int)
    b = np.asarray(labels_b, dtype=int)
    if a.size == 0 or b.size == 0:
        return 0.0
    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    table = np.zeros((len(a_vals), len(b_vals)), dtype=float)
    np.add.at(table, (a_inv, b_inv), 1.0)
    table /= max(table.sum(), 1.0)
    pa = table.sum(axis=1)
    pb = table.sum(axis=0)
    ha = entropy_from_probs(pa)
    hb = entropy_from_probs(pb)
    if ha <= 1e-12 or hb <= 1e-12:
        return 0.0
    nz = table > 0
    denom = pa[:, None] * pb[None, :]
    mi = float((table[nz] * np.log(table[nz] / np.maximum(denom[nz], 1e-12))).sum())
    return float(np.clip(mi / math.sqrt(ha * hb), 0.0, 1.0))


def quantile_bins(values, bins):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=int)
    qs = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    edges = np.unique(np.quantile(values, qs))
    if edges.size == 0:
        return np.zeros(values.shape[0], dtype=int)
    return np.digitize(values, edges, right=False).astype(int)


def curve_landmarks(xs, ys, *, min_gain, q=0.90):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if xs.size == 0:
        return {
            "gain": float("nan"),
            "C_peak": float("nan"),
            "C_q": float("nan"),
            "C_slope": float("nan"),
            "max_slope": float("nan"),
            "detectable": 0,
        }

    ymin = float(np.min(ys))
    ymax = float(np.max(ys))
    gain = ymax - ymin
    c_peak = float(xs[int(np.argmax(ys))])
    c_q = float("nan")
    if gain >= min_gain:
        threshold = ymin + q * gain
        for x, y in zip(xs, ys):
            if y >= threshold:
                c_q = float(x)
                break

    c_slope = float("nan")
    max_slope = float("nan")
    if xs.size >= 2:
        dx = np.diff(xs)
        dy = np.diff(ys)
        valid = dx > 1e-12
        if valid.any():
            slopes = dy[valid] / dx[valid]
            mids = 0.5 * (xs[:-1][valid] + xs[1:][valid])
            best = int(np.argmax(slopes))
            max_slope = float(slopes[best])
            if gain >= min_gain and max_slope > 0:
                c_slope = float(mids[best])

    return {
        "gain": float(gain),
        "C_peak": c_peak,
        "C_q": c_q,
        "C_slope": c_slope,
        "max_slope": max_slope,
        "detectable": int(gain >= min_gain),
    }


def structure_metrics(adapter, *, variant, seed, K, C, split, n):
    key = (variant, seed, int(K), round(float(C), 8), split, int(n))
    if key in adapter._structure_cache:
        return adapter._structure_cache[key]

    model = adapter.model(variant=variant, seed=seed)
    data_split = adapter.split_to_dataset(split) if split != "train_mixed" else "train_mixed"
    x, y = exp.generate_medium_regime(int(n), exp.T, float(C), seed + 70000 + int(31 * C), split=data_split)
    model.eval()
    with torch.no_grad():
        z = model.state(x)

    z = z.detach()
    flat = z.reshape(-1, z.shape[-1])
    mu = flat.mean(dim=0, keepdim=True)
    sigma = flat.std(dim=0, keepdim=True).clamp_min(1e-6)
    z = (z - mu) / sigma
    cur = z[:, :-1, :].reshape(-1, z.shape[-1]).contiguous()
    nxt = z[:, 1:, :].reshape(-1, z.shape[-1]).contiguous()

    centers, assign_cur = exp.kmeans(cur, int(K), seed + 71000 + int(37 * C) + int(K), steps=15)
    assign_next = torch.cdist(nxt, centers).argmin(dim=1)

    counts = torch.bincount(assign_cur, minlength=int(K)).float()
    p = counts / counts.sum().clamp_min(1.0)
    active = p[p > 0]
    usage_entropy = float((-(active * active.log()).sum() / math.log(max(int(K), 2))).item()) if active.numel() else 0.0
    rank_norm = float(exp.effective_rank(cur) / cur.shape[-1])
    rank_score = 0.70 * usage_entropy + 0.30 * rank_norm

    pair = assign_cur * int(K) + assign_next
    trans = torch.bincount(pair, minlength=int(K) * int(K)).float().reshape(int(K), int(K))
    row_sum = trans.sum(dim=1, keepdim=True)
    cond = trans / row_sum.clamp_min(1.0)
    row_prob = row_sum.squeeze(1) / trans.sum().clamp_min(1.0)
    cond_entropy = -(cond.clamp_min(1e-12) * cond.clamp_min(1e-12).log()).sum(dim=1)
    weighted_cond_entropy = float((row_prob * cond_entropy).sum().item())
    trans_pred = float(np.clip(1.0 - weighted_cond_entropy / math.log(max(int(K), 2)), 0.0, 1.0))

    future_y = y[:, 1:].reshape(-1).detach().cpu().numpy()
    future_bins = quantile_bins(future_y, min(8, max(2, int(K))))
    assign_cur_np = assign_cur.detach().cpu().numpy()
    future_mi = normalized_mi(assign_cur_np, future_bins)
    dyn_score = 0.70 * trans_pred + 0.30 * future_mi

    g = torch.Generator(device=assign_next.device)
    g.manual_seed(seed + 73000 + int(41 * C) + int(K))
    perm_next = torch.randperm(assign_next.numel(), generator=g, device=assign_next.device)
    assign_next_null = assign_next[perm_next]
    pair_null = assign_cur * int(K) + assign_next_null
    trans_null = torch.bincount(pair_null, minlength=int(K) * int(K)).float().reshape(int(K), int(K))
    null_row_sum = trans_null.sum(dim=1, keepdim=True)
    null_cond = trans_null / null_row_sum.clamp_min(1.0)
    null_row_prob = null_row_sum.squeeze(1) / trans_null.sum().clamp_min(1.0)
    null_cond_entropy = -(null_cond.clamp_min(1e-12) * null_cond.clamp_min(1e-12).log()).sum(dim=1)
    null_weighted_cond_entropy = float((null_row_prob * null_cond_entropy).sum().item())
    trans_pred_null = float(np.clip(1.0 - null_weighted_cond_entropy / math.log(max(int(K), 2)), 0.0, 1.0))

    rng = np.random.default_rng(seed + 74000 + int(43 * C) + int(K))
    future_mi_null = normalized_mi(assign_cur_np, rng.permutation(future_bins))
    dyn_score_null = 0.70 * trans_pred_null + 0.30 * future_mi_null

    out = {
        "usage_entropy": usage_entropy,
        "rank_norm": rank_norm,
        "rank_score": float(rank_score),
        "trans_pred": trans_pred,
        "future_mi": float(future_mi),
        "dyn_score": float(dyn_score),
        "trans_pred_null": trans_pred_null,
        "future_mi_null": float(future_mi_null),
        "dyn_score_null": float(dyn_score_null),
        "dyn_score_delta": float(dyn_score - dyn_score_null),
    }
    adapter._structure_cache[key] = out
    return out


def run_gate(adapter, args):
    raw = []
    for split in args.splits:
        for variant in args.variants:
            for seed in args.seeds:
                C = adapter.estimate_c_star(variant=variant, seed=seed, split=split)
                b = adapter.eval_baselines(split=split, seed=seed, variant=variant, C=C)
                mse_baseline, baseline_type = gated_baseline(b["mse_random"], b["mse_majority"])
                ratio = full_gain_ratio(mse_baseline, b["mse_full"])
                raw.append(
                    {
                        "split": split,
                        "variant": variant,
                        "seed": seed,
                        "C_star": C,
                        "mse_random": b["mse_random"],
                        "mse_majority": b["mse_majority"],
                        "mse_baseline": mse_baseline,
                        "baseline_type": baseline_type,
                        "mse_full": b["mse_full"],
                        "full_gain_ratio": ratio,
                        "low_signal": int(ratio < args.min_full_gain),
                    }
                )

    summary = []
    for split in args.splits:
        for variant in args.variants:
            rows = [r for r in raw if r["split"] == split and r["variant"] == variant]
            summary.append(
                {
                    "split": split,
                    "variant": variant,
                    "n": len(rows),
                    "full_gain_ratio_mean": mean([r["full_gain_ratio"] for r in rows]),
                    "full_gain_ratio_sem": sem([r["full_gain_ratio"] for r in rows]),
                    "low_signal_rate": mean([r["low_signal"] for r in rows]),
                }
            )

    write_csv(os.path.join(args.outdir, "ood_gate.csv"), raw)
    write_csv(os.path.join(args.outdir, "ood_gate_summary.csv"), summary)
    print_table("Gate summary", summary)


def run_d2(adapter, args):
    raw = []
    for split in args.splits:
        for variant in args.variants:
            for seed in args.seeds:
                C = adapter.estimate_c_star(variant=variant, seed=seed, split=split)
                for K in args.Ks:
                    for bottleneck in BOTTLENECKS:
                        out = adapter.train_and_eval(
                            split=split,
                            variant=variant,
                            seed=seed,
                            K=K,
                            bottleneck=bottleneck,
                            C=C,
                            epochs=args.epochs,
                        )
                        raw.append(
                            {
                                "split": split,
                                "variant": variant,
                                "seed": seed,
                                "K": K,
                                "bottleneck": bottleneck,
                                "C": C,
                                **out,
                            }
                        )

    summary = []
    for split in args.splits:
        for variant in args.variants:
            for K in args.Ks:
                for bottleneck in BOTTLENECKS:
                    rows = [
                        r
                        for r in raw
                        if r["split"] == split and r["variant"] == variant and r["K"] == K and r["bottleneck"] == bottleneck
                    ]
                    summary.append(
                        {
                            "split": split,
                            "variant": variant,
                            "K": K,
                            "bottleneck": bottleneck,
                            "U_mean": mean([r["U"] for r in rows]),
                            "U_sem": sem([r["U"] for r in rows]),
                            "low_signal_rate": mean([r["low_signal"] for r in rows]),
                        }
                    )

    thresholds = []
    for split in args.splits:
        for variant in args.variants:
            for bottleneck in BOTTLENECKS:
                rows = [r for r in summary if r["split"] == split and r["variant"] == variant and r["bottleneck"] == bottleneck]
                rows.sort(key=lambda r: r["K"])
                k90 = next((r["K"] for r in rows if r["U_mean"] >= 0.90), None)
                k95 = next((r["K"] for r in rows if r["U_mean"] >= 0.95), None)
                thresholds.append({"split": split, "variant": variant, "bottleneck": bottleneck, "K90": k90, "K95": k95})

    write_csv(os.path.join(args.outdir, "d2_raw.csv"), raw)
    write_csv(os.path.join(args.outdir, "d2_summary.csv"), summary)
    write_csv(os.path.join(args.outdir, "d2_thresholds.csv"), thresholds)
    print_table("D2 summary", summary)


def run_d4(adapter, args):
    raw = []
    for split in args.splits:
        for variant in args.variants:
            for seed in args.seeds:
                for K in args.Ks:
                    cstar_k = K if args.budget_specific_cstar else None
                    C_star = adapter.estimate_c_star(variant=variant, seed=seed, K=cstar_k, split=split)
                    for rel_C in args.rel_Cs:
                        C = rel_C * C_star
                        out = adapter.train_and_eval(
                            split=split,
                            variant=variant,
                            seed=seed,
                            K=K,
                            bottleneck="discrete",
                            C=C,
                            epochs=args.epochs,
                        )
                        raw.append(
                            {
                                "split": split,
                                "variant": variant,
                                "seed": seed,
                                "K": K,
                                "rel_C": rel_C,
                                "C": C,
                                "C_star": C_star,
                                **out,
                            }
                        )

    curve = []
    for split in args.splits:
        for variant in args.variants:
            for K in args.Ks:
                for rel_C in args.rel_Cs:
                    rows = [r for r in raw if r["split"] == split and r["variant"] == variant and r["K"] == K and r["rel_C"] == rel_C]
                    curve.append(
                        {
                            "split": split,
                            "variant": variant,
                            "K": K,
                            "rel_C": rel_C,
                            "U_mean": mean([r["U"] for r in rows]),
                            "U_sem": sem([r["U"] for r in rows]),
                            "low_signal_rate": mean([r["low_signal"] for r in rows]),
                        }
                    )

    alignment = []
    for split in args.splits:
        for variant in args.variants:
            for K in args.Ks:
                by_seed = {}
                for r in raw:
                    if r["split"] == split and r["variant"] == variant and r["K"] == K:
                        by_seed.setdefault(r["seed"], []).append(r)

                A_vals, best_vals, near_vals, slope_peaks = [], [], [], []
                for seed, rows in by_seed.items():
                    rows = sorted(rows, key=lambda r: r["rel_C"])
                    by_rel = {r["rel_C"]: r["U"] for r in rows}
                    if 0.5 in by_rel and 1.0 in by_rel:
                        A_vals.append(by_rel[1.0] - by_rel[0.5])
                    best_rel = max(rows, key=lambda r: r["U"])["rel_C"]
                    best_vals.append(best_rel)
                    near_vals.append(int(0.75 <= best_rel <= 1.5))
                    if len(rows) >= 2:
                        slopes = []
                        for a, b in zip(rows[:-1], rows[1:]):
                            denom = max(b["rel_C"] - a["rel_C"], 1e-8)
                            slopes.append(((b["U"] - a["U"]) / denom, 0.5 * (a["rel_C"] + b["rel_C"])))
                        slope_peaks.append(max(slopes, key=lambda x: x[0])[1])

                alignment.append(
                    {
                        "split": split,
                        "variant": variant,
                        "K": K,
                        "A_mean": mean(A_vals),
                        "A_sem": sem(A_vals),
                        "best_rel_C_mean": mean(best_vals),
                        "best_rel_C_median": float(stats.median(best_vals)) if best_vals else float("nan"),
                        "aligned_near_rate": mean(near_vals),
                        "slope_peak_rel_C_mean": mean(slope_peaks),
                    }
                )

    write_csv(os.path.join(args.outdir, "d4_raw.csv"), raw)
    write_csv(os.path.join(args.outdir, "d4_curve_summary.csv"), curve)
    write_csv(os.path.join(args.outdir, "d4_alignment.csv"), alignment)
    print_table("D4 alignment", alignment)


def finite_abs_diff(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(abs(a - b))


def finite_less(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return int(a < b)


def finite_dyn_closer(c_rank, c_dyn, c_util):
    if not (np.isfinite(c_rank) and np.isfinite(c_dyn) and np.isfinite(c_util)):
        return float("nan")
    return int(abs(c_dyn - c_util) < abs(c_rank - c_util))


def run_d4_repair(adapter, args):
    raw = []
    for split in args.splits:
        tag = split.upper()
        for variant in args.variants:
            for seed in args.seeds:
                for K in args.Ks:
                    for C in args.C_grid:
                        rows_by_k = adapter.eval_all_budgets(split=split, variant=variant, seed=seed, C=C)
                        budget = rows_by_k[int(K)]
                        sm = structure_metrics(
                            adapter,
                            variant=variant,
                            seed=seed,
                            K=K,
                            C=C,
                            split=args.structure_split,
                            n=args.structure_n,
                        )
                        raw.append(
                            {
                                "split": split,
                                "structure_split": args.structure_split,
                                "variant": variant,
                                "seed": seed,
                                "K": K,
                                "C": float(C),
                                "U_symbol": float(budget[f"U_discrete_{tag}"]),
                                "U_cont_log2": float(budget[f"U_continuous_log2_{tag}"]),
                                "U_cont_K": float(budget[f"U_continuous_K_{tag}"]),
                                "U_random_symbol": float(budget[f"U_random_symbol_{tag}"]),
                                "mse_symbol": float(budget[f"mse_discrete_{tag}"]),
                                "mse_cont_log2": float(budget[f"mse_continuous_log2_{tag}"]),
                                "mse_cont_K": float(budget[f"mse_continuous_K_{tag}"]),
                                "mse_random": float(budget[f"mse_random_{tag}"]),
                                "mse_full": float(budget[f"mse_full_{tag}"]),
                                "low_signal": int(budget[f"low_signal_{tag}"]),
                                **sm,
                            }
                        )

    curve = []
    for split in args.splits:
        for variant in args.variants:
            for K in args.Ks:
                for C in args.C_grid:
                    rows = [r for r in raw if r["split"] == split and r["variant"] == variant and r["K"] == K and r["C"] == float(C)]
                    curve.append(
                        {
                            "split": split,
                            "variant": variant,
                            "K": K,
                            "C": float(C),
                            "U_symbol_mean": mean([r["U_symbol"] for r in rows]),
                            "U_symbol_sem": sem([r["U_symbol"] for r in rows]),
                            "U_cont_log2_mean": mean([r["U_cont_log2"] for r in rows]),
                            "U_cont_K_mean": mean([r["U_cont_K"] for r in rows]),
                            "rank_score_mean": mean([r["rank_score"] for r in rows]),
                            "dyn_score_mean": mean([r["dyn_score"] for r in rows]),
                            "dyn_score_null_mean": mean([r["dyn_score_null"] for r in rows]),
                            "dyn_score_delta_mean": mean([r["dyn_score_delta"] for r in rows]),
                            "trans_pred_mean": mean([r["trans_pred"] for r in rows]),
                            "trans_pred_null_mean": mean([r["trans_pred_null"] for r in rows]),
                            "future_mi_mean": mean([r["future_mi"] for r in rows]),
                            "future_mi_null_mean": mean([r["future_mi_null"] for r in rows]),
                            "low_signal_rate": mean([r["low_signal"] for r in rows]),
                        }
                    )

    landmarks = []
    for split in args.splits:
        for variant in args.variants:
            for K in args.Ks:
                by_seed = {}
                for r in raw:
                    if r["split"] == split and r["variant"] == variant and r["K"] == K:
                        by_seed.setdefault(r["seed"], []).append(r)

                for seed, rows in by_seed.items():
                    rows = sorted(rows, key=lambda r: r["C"])
                    cs = [r["C"] for r in rows]
                    rank_lm = curve_landmarks(cs, [r["rank_score"] for r in rows], min_gain=args.min_structure_gain)
                    dyn_lm = curve_landmarks(cs, [r["dyn_score"] for r in rows], min_gain=args.min_structure_gain)
                    dyn_null_lm = curve_landmarks(cs, [r["dyn_score_null"] for r in rows], min_gain=args.min_structure_gain)
                    dyn_delta_lm = curve_landmarks(cs, [r["dyn_score_delta"] for r in rows], min_gain=args.min_structure_gain)
                    util_lm = curve_landmarks(cs, [r["U_symbol"] for r in rows], min_gain=args.min_utility_gain)
                    cont_lm = curve_landmarks(cs, [r["U_cont_K"] for r in rows], min_gain=args.min_utility_gain)
                    log2_lm = curve_landmarks(cs, [r["U_cont_log2"] for r in rows], min_gain=args.min_utility_gain)

                    landmarks.append(
                        {
                            "split": split,
                            "variant": variant,
                            "seed": seed,
                            "K": K,
                            "rank_gain": rank_lm["gain"],
                            "dyn_gain": dyn_lm["gain"],
                            "dyn_null_gain": dyn_null_lm["gain"],
                            "dyn_delta_gain": dyn_delta_lm["gain"],
                            "symbol_gain": util_lm["gain"],
                            "contK_gain": cont_lm["gain"],
                            "cont_log2_gain": log2_lm["gain"],
                            "rank_detectable": rank_lm["detectable"],
                            "dyn_detectable": dyn_lm["detectable"],
                            "dyn_null_detectable": dyn_null_lm["detectable"],
                            "dyn_delta_detectable": dyn_delta_lm["detectable"],
                            "symbol_detectable": util_lm["detectable"],
                            "contK_detectable": cont_lm["detectable"],
                            "C_rank_slope": rank_lm["C_slope"],
                            "C_dyn_slope": dyn_lm["C_slope"],
                            "C_dyn_null_slope": dyn_null_lm["C_slope"],
                            "C_dyn_delta_slope": dyn_delta_lm["C_slope"],
                            "C_util_slope": util_lm["C_slope"],
                            "C_contK_slope": cont_lm["C_slope"],
                            "C_rank_q90": rank_lm["C_q"],
                            "C_dyn_q90": dyn_lm["C_q"],
                            "C_dyn_null_q90": dyn_null_lm["C_q"],
                            "C_dyn_delta_q90": dyn_delta_lm["C_q"],
                            "C_util_q90": util_lm["C_q"],
                            "C_contK_q90": cont_lm["C_q"],
                            "C_rank_peak": rank_lm["C_peak"],
                            "C_dyn_peak": dyn_lm["C_peak"],
                            "C_dyn_null_peak": dyn_null_lm["C_peak"],
                            "C_dyn_delta_peak": dyn_delta_lm["C_peak"],
                            "C_util_peak": util_lm["C_peak"],
                            "rank_to_util_slope_err": finite_abs_diff(rank_lm["C_slope"], util_lm["C_slope"]),
                            "dyn_to_util_slope_err": finite_abs_diff(dyn_lm["C_slope"], util_lm["C_slope"]),
                            "dyn_null_to_util_slope_err": finite_abs_diff(dyn_null_lm["C_slope"], util_lm["C_slope"]),
                            "dyn_delta_to_util_slope_err": finite_abs_diff(dyn_delta_lm["C_slope"], util_lm["C_slope"]),
                            "rank_to_util_q90_err": finite_abs_diff(rank_lm["C_q"], util_lm["C_q"]),
                            "dyn_to_util_q90_err": finite_abs_diff(dyn_lm["C_q"], util_lm["C_q"]),
                            "dyn_null_to_util_q90_err": finite_abs_diff(dyn_null_lm["C_q"], util_lm["C_q"]),
                            "dyn_delta_to_util_q90_err": finite_abs_diff(dyn_delta_lm["C_q"], util_lm["C_q"]),
                            "dyn_closer_slope": finite_dyn_closer(rank_lm["C_slope"], dyn_lm["C_slope"], util_lm["C_slope"]),
                            "dyn_closer_q90": finite_dyn_closer(rank_lm["C_q"], dyn_lm["C_q"], util_lm["C_q"]),
                            "dyn_beats_null_slope": finite_dyn_closer(dyn_null_lm["C_slope"], dyn_lm["C_slope"], util_lm["C_slope"]),
                            "dyn_beats_null_q90": finite_dyn_closer(dyn_null_lm["C_q"], dyn_lm["C_q"], util_lm["C_q"]),
                            "dyn_delta_beats_null_q90": finite_dyn_closer(dyn_null_lm["C_q"], dyn_delta_lm["C_q"], util_lm["C_q"]),
                            "rank_before_util_slope": finite_less(rank_lm["C_slope"], util_lm["C_slope"]),
                            "dyn_before_util_slope": finite_less(dyn_lm["C_slope"], util_lm["C_slope"]),
                        }
                    )

    alignment = []
    for split in args.splits:
        for variant in args.variants:
            for K in args.Ks:
                rows = [r for r in landmarks if r["split"] == split and r["variant"] == variant and r["K"] == K]
                slope_rows = [r for r in rows if np.isfinite(r["C_util_slope"])]
                q_rows = [r for r in rows if np.isfinite(r["C_util_q90"])]
                alignment.append(
                    {
                        "split": split,
                        "variant": variant,
                        "K": K,
                        "n": len(rows),
                        "symbol_detect_rate": mean([r["symbol_detectable"] for r in rows]),
                        "dyn_detect_rate": mean([r["dyn_detectable"] for r in rows]),
                        "dyn_null_detect_rate": mean([r["dyn_null_detectable"] for r in rows]),
                        "dyn_delta_detect_rate": mean([r["dyn_delta_detectable"] for r in rows]),
                        "rank_detect_rate": mean([r["rank_detectable"] for r in rows]),
                        "symbol_gain_mean": mean([r["symbol_gain"] for r in rows]),
                        "contK_gain_mean": mean([r["contK_gain"] for r in rows]),
                        "dyn_gain_mean": mean([r["dyn_gain"] for r in rows]),
                        "dyn_null_gain_mean": mean([r["dyn_null_gain"] for r in rows]),
                        "dyn_delta_gain_mean": mean([r["dyn_delta_gain"] for r in rows]),
                        "rank_gain_mean": mean([r["rank_gain"] for r in rows]),
                        "dyn_to_util_slope_err_mean": mean([r["dyn_to_util_slope_err"] for r in slope_rows if np.isfinite(r["dyn_to_util_slope_err"])]),
                        "dyn_null_to_util_slope_err_mean": mean([r["dyn_null_to_util_slope_err"] for r in slope_rows if np.isfinite(r["dyn_null_to_util_slope_err"])]),
                        "dyn_delta_to_util_slope_err_mean": mean([r["dyn_delta_to_util_slope_err"] for r in slope_rows if np.isfinite(r["dyn_delta_to_util_slope_err"])]),
                        "rank_to_util_slope_err_mean": mean([r["rank_to_util_slope_err"] for r in slope_rows if np.isfinite(r["rank_to_util_slope_err"])]),
                        "dyn_closer_slope_rate": mean([r["dyn_closer_slope"] for r in slope_rows if np.isfinite(r["dyn_closer_slope"])]),
                        "dyn_beats_null_slope_rate": mean([r["dyn_beats_null_slope"] for r in slope_rows if np.isfinite(r["dyn_beats_null_slope"])]),
                        "dyn_before_util_slope_rate": mean([r["dyn_before_util_slope"] for r in slope_rows if np.isfinite(r["dyn_before_util_slope"])]),
                        "dyn_to_util_q90_err_mean": mean([r["dyn_to_util_q90_err"] for r in q_rows if np.isfinite(r["dyn_to_util_q90_err"])]),
                        "dyn_null_to_util_q90_err_mean": mean([r["dyn_null_to_util_q90_err"] for r in q_rows if np.isfinite(r["dyn_null_to_util_q90_err"])]),
                        "dyn_delta_to_util_q90_err_mean": mean([r["dyn_delta_to_util_q90_err"] for r in q_rows if np.isfinite(r["dyn_delta_to_util_q90_err"])]),
                        "rank_to_util_q90_err_mean": mean([r["rank_to_util_q90_err"] for r in q_rows if np.isfinite(r["rank_to_util_q90_err"])]),
                        "dyn_closer_q90_rate": mean([r["dyn_closer_q90"] for r in q_rows if np.isfinite(r["dyn_closer_q90"])]),
                        "dyn_beats_null_q90_rate": mean([r["dyn_beats_null_q90"] for r in q_rows if np.isfinite(r["dyn_beats_null_q90"])]),
                        "dyn_delta_beats_null_q90_rate": mean([r["dyn_delta_beats_null_q90"] for r in q_rows if np.isfinite(r["dyn_delta_beats_null_q90"])]),
                    }
                )

    write_csv(os.path.join(args.outdir, "d4_repair_raw.csv"), raw)
    write_csv(os.path.join(args.outdir, "d4_repair_curve_summary.csv"), curve)
    write_csv(os.path.join(args.outdir, "d4_repair_landmarks.csv"), landmarks)
    write_csv(os.path.join(args.outdir, "d4_repair_alignment.csv"), alignment)
    print_table("D4 repair alignment", alignment)


def print_table(title, rows, max_rows=40):
    print(f"\n{title}", flush=True)
    for row in rows[:max_rows]:
        print("  " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more rows", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gate", "d2", "d4", "d4_repair"], required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--splits", nargs="+", default=["id"])
    p.add_argument("--variants", nargs="+", default=["original", "pre_softmax"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--Ks", nargs="+", type=int, default=[4, 8, 16, 64])
    p.add_argument("--rel-Cs", dest="rel_Cs", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    p.add_argument("--C-grid", dest="C_grid", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 24.0])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--min-full-gain", type=float, default=0.05)
    p.add_argument("--min-utility-gain", type=float, default=0.10)
    p.add_argument("--min-structure-gain", type=float, default=0.02)
    p.add_argument("--structure-split", default="train_mixed")
    p.add_argument("--structure-n", type=int, default=384)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--budget-specific-cstar", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = exp.set_device(args.device)
    print(f"[device] {device}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    cfg = ProtocolConfig(epochs=args.epochs, min_full_gain=args.min_full_gain)
    adapter = MyAdapter(cfg)
    if args.mode == "gate":
        run_gate(adapter, args)
    elif args.mode == "d2":
        run_d2(adapter, args)
    elif args.mode == "d4":
        run_d4(adapter, args)
    elif args.mode == "d4_repair":
        run_d4_repair(adapter, args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
