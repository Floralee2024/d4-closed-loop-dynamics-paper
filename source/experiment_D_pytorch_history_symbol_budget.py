"""
Experiment D: symbol budget x decision utility on a history-dependent regime task.

This is the PyTorch version of D1/D2/D4:
  D1: medium_regime task with multiple regimes, history dependence,
      nonlinear mode switches, and spurious train/test shift.
  D2: discrete symbols vs continuous bottleneck controls:
      log2(K)-dimensional and K-dimensional PCA bottlenecks.
  D4: unsupervised C* estimated from learned state structure, then utility
      evaluated at relative C/C* without moving C* using downstream utility.
  ID/OOD split: utility is reported separately for in-distribution test data and
      OOD spurious-shift test data.

Smoke test:
  python experiment_D_pytorch_history_symbol_budget.py --seed-count 2 --epochs 40

Main-ish run:
  python experiment_D_pytorch_history_symbol_budget.py --seed-count 20 --epochs 200
"""

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = os.environ.get("BIR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
OUT = r"D:\newplan\experiment_D_history_symbol_budget"

OBS_DIM = 6
HIDDEN_DIM = 32
T = 18
TRAIN_N = 768
EVAL_TRAIN_N = 512
EVAL_TEST_N = 512

KS = [1, 2, 4, 8, 16, 64]
REL_CS = [0.0, 0.5, 1.0, 1.5]
VARIANTS = ["original", "pre_softmax"]
C_SCAN = [0, 2, 4, 6, 8, 10, 12, 16, 20]


def resolve_device(device):
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment cannot see a CUDA device.")
    return device


def set_device(device):
    global DEVICE
    DEVICE = resolve_device(device)
    return DEVICE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_medium_regime(n, t_steps, C, seed, split="train", device=None):
    """History-dependent regime task.

    There are 4 hidden regimes. Regime transitions depend on a smoothed history
    variable and nonlinear gates. A spurious feature is correlated with regime
    under ID evaluation, but training can mix several spurious mappings and
    apply dropout. The target depends on the true regime and stable history
    features, not on the spurious feature.
    """

    if device is None:
        device = DEVICE
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    mode = torch.randint(0, 4, (n,), generator=g, device=device)
    levels = torch.tensor([-1.0, -0.25, 0.35, 1.05], device=device)
    init_levels = torch.tensor([-1.1, -0.35, 0.35, 1.1], device=device)
    hist = init_levels[mode] + 0.08 * torch.randn(n, generator=g, device=device)
    momentum = torch.zeros(n, device=device)

    obs, target = [], []
    prev_b = torch.randn(n, generator=g, device=device) * 0.15

    for _ in range(t_steps):
        exog = torch.randn(n, 4, generator=g, device=device)
        b_diff = torch.tanh(0.55 * prev_b + 0.35 * exog[:, 0] + 0.08 * C * exog[:, 1])
        m_diff = torch.tanh(0.60 * hist + 0.45 * exog[:, 2] + 0.12 * C * b_diff)

        momentum = 0.70 * momentum + 0.30 * b_diff
        hist = 0.90 * hist + 0.10 * (init_levels[mode] + 0.40 * m_diff + 0.30 * momentum)

        switch_gate = torch.sin(2.6 * hist + 0.18 * C) + 0.55 * torch.tanh(2.0 * m_diff)
        mode = torch.where(switch_gate > 1.05, (mode + 1) % 4, mode)
        mode = torch.where(switch_gate < -1.05, (mode + 3) % 4, mode)

        raw_spur = exog[:, 3]
        base_spur = torch.where(mode < 2, raw_spur.abs(), -raw_spur.abs())

        if split in ("train", "id"):
            signed_spur = base_spur
        elif split == "train_mixed":
            env = torch.rand(n, generator=g, device=device)
            random_sign = torch.where(
                torch.rand(n, generator=g, device=device) > 0.5,
                torch.ones(n, device=device),
                -torch.ones(n, device=device),
            )
            signed_spur = torch.where(env < 0.70, base_spur, torch.where(env < 0.85, -base_spur, raw_spur.abs() * random_sign))
            keep = (torch.rand(n, generator=g, device=device) > 0.35).float()
            signed_spur = signed_spur * keep
        elif split in ("ood", "ood_inverted", "test"):
            signed_spur = -base_spur
        elif split == "ood_mild":
            flip = torch.rand(n, generator=g, device=device) < 0.35
            signed_spur = torch.where(flip, -base_spur, base_spur)
        elif split == "ood_random":
            random_sign = torch.where(
                torch.rand(n, generator=g, device=device) > 0.5,
                torch.ones(n, device=device),
                -torch.ones(n, device=device),
            )
            signed_spur = raw_spur.abs() * random_sign
        else:
            raise ValueError(f"Unknown split: {split}")

        y = levels[mode] + 0.10 * torch.tanh(m_diff) + 0.05 * b_diff
        x = torch.stack(
            [
                hist + 0.06 * torch.randn(n, generator=g, device=device),
                momentum,
                m_diff,
                b_diff,
                torch.sin(2.6 * hist + 0.18 * C),
                0.10 * signed_spur,
            ],
            dim=-1,
        )
        obs.append(x)
        target.append(y)
        prev_b = b_diff

    return torch.stack(obs, dim=1), torch.stack(target, dim=1)


def standardize_pair(x_train, x_test):
    mu = x_train.reshape(-1, x_train.shape[-1]).mean(dim=0)
    sigma = x_train.reshape(-1, x_train.shape[-1]).std(dim=0).clamp_min(1e-6)
    return (x_train - mu) / sigma, (x_test - mu) / sigma


class HistoryModel(nn.Module):
    def __init__(self, variant="original", obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.variant = variant
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.tau = nn.Parameter(torch.tensor(1.0))

    def state(self, x):
        h, _ = self.gru(x)
        h = self.norm(h)
        if self.variant == "original":
            return h
        if self.variant == "pre_softmax":
            return torch.softmax(F.relu(h), dim=-1)
        if self.variant == "temp_softmax":
            tau = self.tau.abs().clamp_min(1e-3)
            return torch.softmax(h / tau, dim=-1)
        if self.variant == "no_softmax":
            return h / (h.abs().sum(dim=-1, keepdim=True) + 1e-8)
        raise ValueError(f"Unknown variant: {self.variant}")

    def forward(self, x):
        z = self.state(x)
        return self.head(z).squeeze(-1), z


def train_model(variant, seed, epochs, batch_size=128):
    set_seed(seed)
    model = HistoryModel(variant=variant).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()

    for ep in range(epochs):
        C = float(torch.rand(1).item() * 12.0)
        x, y = generate_medium_regime(TRAIN_N, T, C, seed * 10000 + ep, split="train_mixed")
        perm = torch.randperm(TRAIN_N)
        for start in range(0, TRAIN_N, batch_size):
            idx = perm[start : start + batch_size]
            pred, _ = model(x[idx])
            loss = F.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def flatten_states(model, C, seed, split, n):
    x, y = generate_medium_regime(n, T, C, seed, split=split)
    model.eval()
    with torch.no_grad():
        _, z = model(x)
    return z.reshape(-1, z.shape[-1]).detach(), y.reshape(-1).detach()


def kmeans(x, k, seed, steps=25):
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))
    idx = torch.randperm(x.shape[0], generator=g, device=x.device)[:k]
    centers = x[idx].clone()
    assign = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(steps):
        assign = torch.cdist(x, centers).argmin(dim=1)
        new_centers = centers.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = x[mask].mean(dim=0)
        centers = new_centers
    assign = torch.cdist(x, centers).argmin(dim=1)
    return centers, assign


def cluster_predict_mse(x_train, y_train, x_test, y_test, k, seed):
    centers, assign_train = kmeans(x_train, k, seed)
    global_mean = y_train.mean()
    values = torch.full((k,), global_mean.item(), device=x_train.device)
    for j in range(k):
        mask = assign_train == j
        if mask.any():
            values[j] = y_train[mask].mean()
    assign_test = torch.cdist(x_test, centers).argmin(dim=1)
    pred = values[assign_test]
    return F.mse_loss(pred, y_test).item()


def random_symbol_mse(y_train, y_test, k, seed):
    g = torch.Generator(device=y_train.device)
    g.manual_seed(int(seed))
    a_train = torch.randint(0, k, (y_train.numel(),), generator=g, device=y_train.device)
    a_test = torch.randint(0, k, (y_test.numel(),), generator=g, device=y_test.device)
    global_mean = y_train.mean()
    values = torch.full((k,), global_mean.item(), device=y_train.device)
    for j in range(k):
        mask = a_train == j
        if mask.any():
            values[j] = y_train[mask].mean()
    return F.mse_loss(values[a_test], y_test).item()


def ridge_mse(x_train, y_train, x_test, y_test, lam=1e-3):
    ones_train = torch.ones(x_train.shape[0], 1, device=x_train.device)
    ones_test = torch.ones(x_test.shape[0], 1, device=x_test.device)
    xt = torch.cat([ones_train, x_train], dim=1)
    xv = torch.cat([ones_test, x_test], dim=1)
    # Solve the intercept-unregularized ridge problem through an augmented
    # least-squares system.  Forming X.T @ X and calling solve can fail on
    # CPU when symbolic features are rank-deficient, even though the intended
    # ridge objective is well-defined for the non-intercept coefficients.
    penalty = torch.zeros(xt.shape[1], xt.shape[1], device=x_train.device)
    penalty[1:, 1:] = torch.eye(xt.shape[1] - 1, device=x_train.device) * math.sqrt(lam)
    augmented_x = torch.cat([xt, penalty], dim=0)
    augmented_y = torch.cat([y_train, torch.zeros(penalty.shape[0], device=y_train.device)], dim=0)
    w = torch.linalg.lstsq(augmented_x, augmented_y).solution
    pred = xv @ w
    return F.mse_loss(pred, y_test).item()


def pca_project_pair(x_train, x_test, dim):
    mu = x_train.mean(dim=0, keepdim=True)
    xc = x_train - mu
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    basis = vh[:dim].T
    return (x_train - mu) @ basis, (x_test - mu) @ basis


def effective_rank(x):
    xc = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(xc)
    ev = (s * s).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()).item())


def unsup_metrics(model, C, k, seed):
    z, _ = flatten_states(model, C, seed, split="train", n=256)
    centers, assign = kmeans(z, k, seed + 17, steps=15)
    counts = torch.bincount(assign, minlength=k).float()
    p = counts / counts.sum().clamp_min(1.0)
    p = p[p > 0]
    usage_entropy = float((-(p * p.log()).sum() / math.log(k)).item())
    rank_norm = effective_rank(z) / z.shape[-1]
    return {
        "usage_entropy": usage_entropy,
        "rank_norm": rank_norm,
        "score": 0.70 * usage_entropy + 0.30 * rank_norm,
    }


def find_unsup_c_star(model, seed, k=16):
    best = None
    for C in C_SCAN:
        m = unsup_metrics(model, float(C), k, seed + int(C * 101))
        cur = {"C_star": float(C), **m}
        if best is None or cur["score"] > best["score"]:
            best = cur
        if cur["usage_entropy"] >= 0.72 and cur["score"] >= 0.72:
            return cur
    return best


def eval_budget_for_split(z_train, y_train, z_test, y_test, seed, split_name):
    z_train, z_test = standardize_pair(z_train, z_test)
    mse_random = F.mse_loss(torch.full_like(y_test, y_train.mean()), y_test).item()
    mse_majority = F.mse_loss(torch.full_like(y_test, y_train.median()), y_test).item()

    raw_rows = []
    candidate_full = [ridge_mse(z_train, y_train, z_test, y_test)]
    for k in KS:
        mse_disc = cluster_predict_mse(z_train, y_train, z_test, y_test, k, seed + 100 * k)
        mse_rand_sym = random_symbol_mse(y_train, y_test, k, seed + 200 * k)
        dim_log2 = max(1, min(z_train.shape[-1], math.ceil(math.log2(k))))
        dim_k = max(1, min(z_train.shape[-1], k))
        ct_log2, cv_log2 = pca_project_pair(z_train, z_test, dim_log2)
        ct_k, cv_k = pca_project_pair(z_train, z_test, dim_k)
        mse_cont_log2 = ridge_mse(ct_log2, y_train, cv_log2, y_test)
        mse_cont_k = ridge_mse(ct_k, y_train, cv_k, y_test)
        candidate_full.extend([mse_disc, mse_cont_log2, mse_cont_k])
        raw_rows.append((k, dim_log2, dim_k, mse_disc, mse_cont_log2, mse_cont_k, mse_rand_sym))

    mse_full = min(candidate_full)
    denom = mse_random - mse_full
    low_signal = denom < 1e-6
    denom = max(denom, 1e-6)

    by_k = {}
    for k, dim_log2, dim_k, mse_disc, mse_cont_log2, mse_cont_k, mse_rand_sym in raw_rows:
        def util(mse_value):
            return float(np.clip((mse_random - mse_value) / denom, -0.5, 1.2))

        by_k[k] = {
            f"mse_random_{split_name}": mse_random,
            f"mse_majority_{split_name}": mse_majority,
            f"mse_full_{split_name}": mse_full,
            f"low_signal_{split_name}": int(low_signal),
            f"mse_discrete_{split_name}": mse_disc,
            f"mse_continuous_log2_{split_name}": mse_cont_log2,
            f"mse_continuous_K_{split_name}": mse_cont_k,
            f"U_discrete_{split_name}": util(mse_disc),
            f"U_continuous_log2_{split_name}": util(mse_cont_log2),
            f"U_continuous_K_{split_name}": util(mse_cont_k),
            f"U_random_symbol_{split_name}": util(mse_rand_sym),
            "cont_dim_log2": dim_log2,
            "cont_dim_K": dim_k,
        }
    return by_k


def eval_budget(model, C, seed):
    z_train, y_train = flatten_states(model, C, seed + 1, split="train", n=EVAL_TRAIN_N)
    z_test_id, y_test_id = flatten_states(model, C, seed + 2, split="train", n=EVAL_TEST_N)
    z_test_ood, y_test_ood = flatten_states(model, C, seed + 3, split="ood_inverted", n=EVAL_TEST_N)

    id_by_k = eval_budget_for_split(z_train, y_train, z_test_id, y_test_id, seed + 10000, "ID")
    ood_by_k = eval_budget_for_split(z_train, y_train, z_test_ood, y_test_ood, seed + 20000, "OOD")
    rows = []
    for k in KS:
        merged = {"K": k, **id_by_k[k], **ood_by_k[k]}
        rows.append(
            {
                **merged,
            }
        )
    return rows


def summarize(rows, split_name="ID"):
    groups = {}
    for r in rows:
        key = (r["variant"], r["rel_C"], r["K"])
        groups.setdefault(key, []).append(r)
    summary = []
    for (variant, rel_C, k), rs in sorted(groups.items()):
        vals = np.array([x[f"U_discrete_{split_name}"] for x in rs], dtype=float)
        cont_log2 = np.array([x[f"U_continuous_log2_{split_name}"] for x in rs], dtype=float)
        cont_k = np.array([x[f"U_continuous_K_{split_name}"] for x in rs], dtype=float)
        rand = np.array([x[f"U_random_symbol_{split_name}"] for x in rs], dtype=float)
        summary.append(
            {
                "variant": variant,
                "rel_C": rel_C,
                "K": k,
                "split": split_name,
                "U_discrete_mean": float(vals.mean()),
                "U_discrete_std": float(vals.std()),
                "U_continuous_log2_mean": float(cont_log2.mean()),
                "U_continuous_K_mean": float(cont_k.mean()),
                "U_random_symbol_mean": float(rand.mean()),
                "discrete_minus_cont_log2": float((vals - cont_log2).mean()),
                "discrete_minus_cont_K": float((vals - cont_k).mean()),
                "low_signal_rate": float(np.mean([x[f"low_signal_{split_name}"] for x in rs])),
            }
        )
    return summary


def d4_alignment(summary, split_name="ID"):
    """Held-out validation of unsupervised C*.

    C* is estimated without utility. This reports whether downstream utility at
    C* beats the pre-critical point and whether the curve plateaus after C*.
    """

    rows = []
    variants = sorted({r["variant"] for r in summary})
    for variant in variants:
        for k in KS:
            by_rel = {r["rel_C"]: r for r in summary if r["variant"] == variant and r["K"] == k}
            if 0.5 not in by_rel or 1.0 not in by_rel or 1.5 not in by_rel:
                continue
            u_half = by_rel[0.5]["U_discrete_mean"]
            u_star = by_rel[1.0]["U_discrete_mean"]
            u_after = by_rel[1.5]["U_discrete_mean"]
            best_rel = max(by_rel, key=lambda rel: by_rel[rel]["U_discrete_mean"])
            rows.append(
                {
                    "variant": variant,
                    "K": k,
                    "split": split_name,
                    "A_star_minus_half": float(u_star - u_half),
                    "plateau_after_star": float(u_after - u_star),
                    "best_rel_C": float(best_rel),
                    "best_minus_star": float(by_rel[best_rel]["U_discrete_mean"] - u_star),
                    "aligned_best_near_star": int(best_rel in (1.0, 1.5)),
                }
            )
    return rows


def k_threshold(summary, variant, rel_C, threshold, split_name="ID"):
    rows = [r for r in summary if r["variant"] == variant and r["rel_C"] == rel_C and r["split"] == split_name]
    rows.sort(key=lambda r: r["K"])
    for r in rows:
        if r["U_discrete_mean"] >= threshold:
            return r["K"]
    return None


def run(args):
    device = set_device(args.device)
    print(f"[device] {device}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    all_rows = []

    for variant in args.variants.split(","):
        variant = variant.strip()
        if not variant:
            continue
        for seed in range(args.seed_count):
            print(f"[train] variant={variant} seed={seed}", flush=True)
            model = train_model(variant, seed=seed, epochs=args.epochs)
            c_info = find_unsup_c_star(model, seed=50000 + seed)
            c_star = max(c_info["C_star"], 0.5)
            print(
                f"  C*_unsup={c_star:.3f} entropy={c_info['usage_entropy']:.3f} "
                f"rank={c_info['rank_norm']:.3f}",
                flush=True,
            )
            for rel_C in REL_CS:
                C = rel_C * c_star
                budget_rows = eval_budget(model, C, seed=70000 + 1000 * seed + int(rel_C * 10))
                for row in budget_rows:
                    all_rows.append(
                        {
                            "task": "medium_regime_history",
                            "variant": variant,
                            "seed": seed,
                            "rel_C": rel_C,
                            "C": C,
                            "C_star": c_star,
                            "C_usage_entropy": c_info["usage_entropy"],
                            "C_rank_norm": c_info["rank_norm"],
                            **row,
                        }
                    )

    csv_path = os.path.join(args.out, "experiment_D_pytorch_history_symbol_budget.csv")
    fields = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_id = summarize(all_rows, "ID")
    summary_ood = summarize(all_rows, "OOD")
    summary = summary_id + summary_ood
    d4 = d4_alignment(summary_id, "ID") + d4_alignment(summary_ood, "OOD")
    report = {"summary": summary, "d4_alignment": d4, "K90": [], "K95": []}
    for variant in args.variants.split(","):
        variant = variant.strip()
        for rel_C in REL_CS:
            for split_name in ["ID", "OOD"]:
                report["K90"].append({"variant": variant, "rel_C": rel_C, "split": split_name, "K90": k_threshold(summary, variant, rel_C, 0.90, split_name)})
                report["K95"].append({"variant": variant, "rel_C": rel_C, "split": split_name, "K95": k_threshold(summary, variant, rel_C, 0.95, split_name)})

    json_path = os.path.join(args.out, "experiment_D_pytorch_history_symbol_budget_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nSummary at C/C*=1:", flush=True)
    for variant in args.variants.split(","):
        variant = variant.strip()
        for split_name in ["ID", "OOD"]:
            print(f"  [{split_name}] {variant}", flush=True)
            for k in [4, 8, 16, 64]:
                s = next(r for r in summary if r["variant"] == variant and r["rel_C"] == 1.0 and r["K"] == k and r["split"] == split_name)
                print(
                    f"    K={k:>2d}: disc={s['U_discrete_mean']:.3f} "
                    f"cont_log2={s['U_continuous_log2_mean']:.3f} "
                    f"cont_K={s['U_continuous_K_mean']:.3f} "
                    f"d-log2={s['discrete_minus_cont_log2']:.3f} "
                    f"d-K={s['discrete_minus_cont_K']:.3f} "
                    f"low_signal={s['low_signal_rate']:.2f}",
                    flush=True,
                )
            print(
                f"    K90={k_threshold(summary, variant, 1.0, 0.90, split_name)} "
                f"K95={k_threshold(summary, variant, 1.0, 0.95, split_name)}",
                flush=True,
            )

    print("\nD4 alignment at K=16:", flush=True)
    for variant in args.variants.split(","):
        variant = variant.strip()
        for split_name in ["ID", "OOD"]:
            row = next((r for r in d4 if r["variant"] == variant and r["K"] == 16 and r["split"] == split_name), None)
            if not row:
                continue
            print(
                f"  [{split_name}] {variant:>12s}: A=U(C*)-U(.5C*)={row['A_star_minus_half']:.3f} "
                f"plateau=U(1.5C*)-U(C*)={row['plateau_after_star']:.3f} "
                f"best_rel_C={row['best_rel_C']:.1f}",
                flush=True,
            )
    print(f"\nSaved {csv_path}", flush=True)
    print(f"Saved {json_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    p.add_argument("--seed-count", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--variants", default="original,pre_softmax")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return p.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    run(parse_args())
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min", flush=True)
