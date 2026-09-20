#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D4b: Residual Bypass Desynchronization.

This self-contained GPU-friendly script tests the upgraded D4b hypothesis:
  pure symbolic (residual_strength=0):     C*_rank ~= C*_dyn
  weak residual:                           mild separation
  strong residual / full hybrid (=1.0):    large separation
  ID and OOD ordering are reported separately and never averaged away.

Residual strength is a single continuous knob:
  residual_strength = 0.0  -> pure symbolic, q = gate(C) * symbol
  residual_strength = 1.0  -> full hybrid,   q = gate(C) * symbol + (1-gate(C)) * continuous
  intermediate values      -> q = gate(C) * symbol + residual_strength*(1-gate(C))*continuous

Only q90 C* is used. Slope-peak C* is intentionally omitted because it is noisy.

Outputs:
  - curves.csv: raw R(C), D(C), U(C) curves
  - cstar_summary.csv: per-seed C*_rank_q90, C*_dyn_q90, C*_util_q90 and D4b metrics
  - aggregate.csv: grouped mean/std comparison
  - d4b_by_strength.csv: split-specific residual-strength summary
  - d4b_strength_summary.csv: compact summary over K/split

GPU smoke test:
  python d4b_residual_desync_gpu.py --steps 30 --seeds 0 --Ks 8 --Cs 0 1 3 8 \
    --residual-strengths 0 0.25 1.0 --splits ood_inverted \
    --out-dir runs/d4b_smoke --device cuda --amp auto --tf32 --num-workers 2 --batch-size 256

Recommended run:
  python d4b_residual_desync_gpu.py --steps 300 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --Ks 8 16 --Cs 0 0.5 1 2 3 5 8 12 16 24 --variants original \
    --residual-strengths 0 0.1 0.25 0.5 1.0 --splits id ood_random ood_inverted \
    --device cuda --amp auto --tf32 --batch-size 512 --eval-batch-size 2048 \
    --num-workers 2 --persistent-workers --out-dir runs/d4b_10seed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# -----------------------------
# Basic utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(device)


def configure_gpu(device: torch.device, tf32: bool = False, matmul_precision: str = "high") -> None:
    """Enable common GPU speed knobs. Safe no-op on CPU."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    # TF32 is usually a good speed/accuracy tradeoff for Ampere+ GPUs.
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        pass


def resolve_amp(device: torch.device, amp: str) -> Tuple[bool, Optional[torch.dtype], bool]:
    """Return (amp_enabled, autocast_dtype, scaler_enabled).

    --amp auto chooses bf16 on Ampere+ where supported, otherwise fp16.
    GradScaler is only enabled for fp16 CUDA AMP.
    """
    if device.type != "cuda" or amp == "off":
        return False, None, False
    if amp == "bf16":
        return True, torch.bfloat16, False
    if amp == "fp16":
        return True, torch.float16, True
    # auto
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 8:
        return True, torch.bfloat16, False
    return True, torch.float16, True


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=(device.type == "cuda"),
        num_workers=max(0, int(num_workers)),
    )
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(ds, **kwargs)


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device, non_blocking=(device.type == "cuda"))


def entropy_np(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(p + eps)))


def normalized_mi(counts: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetric normalized mutual information from a contingency table."""
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    pxy = counts / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mi = np.sum(pxy * (np.log(pxy + eps) - np.log(px + eps) - np.log(py + eps)))
    hx = entropy_np(px.squeeze(), eps)
    hy = entropy_np(py.squeeze(), eps)
    denom = math.sqrt(max(hx * hy, eps))
    return float(mi / denom)


def effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank of covariance of representation matrix X [N,D]."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or len(X) < 3:
        return float("nan")
    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(len(X) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    s = eigvals.sum()
    if s <= eps:
        return 0.0
    p = eigvals / s
    return float(np.exp(-np.sum(p * np.log(p + eps))))


def normalize_curve_values(x: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(x), dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def curve_gain(ys: Iterable[float]) -> float:
    ys = np.asarray(list(ys), dtype=np.float64)
    ys = ys[np.isfinite(ys)]
    if len(ys) == 0:
        return float("nan")
    return float(np.nanmax(ys) - ys[0])


def critical_C_threshold(Cs: Iterable[float], ys: Iterable[float], frac: float = 0.90, min_gain: float = 0.0) -> float:
    """First C reaching frac of observed gain from y(C_min) to max_C y(C).

    If observed gain is below min_gain, no meaningful C* is extracted.
    """
    Cs = np.asarray(list(Cs), dtype=np.float64)
    ys = np.asarray(list(ys), dtype=np.float64)
    ok = np.isfinite(Cs) & np.isfinite(ys)
    Cs, ys = Cs[ok], ys[ok]
    if len(Cs) == 0:
        return float("nan")
    order = np.argsort(Cs)
    Cs, ys = Cs[order], ys[order]

    ys_s = ys.copy()
    if len(ys_s) >= 3:
        ys_s[1:-1] = 0.25 * ys[:-2] + 0.5 * ys[1:-1] + 0.25 * ys[2:]

    y0 = ys_s[0]
    ymax = np.nanmax(ys_s)
    gain = ymax - y0
    if gain < min_gain:
        return float("nan")
    threshold = y0 + frac * gain
    idxs = np.where(ys_s >= threshold)[0]
    return float(Cs[idxs[0]]) if len(idxs) else float("nan")

def critical_C_slope(Cs: Iterable[float], ys: Iterable[float]) -> float:
    """C midpoint of maximum positive slope. Useful as a secondary diagnostic."""
    Cs = np.asarray(list(Cs), dtype=np.float64)
    ys = np.asarray(list(ys), dtype=np.float64)
    ok = np.isfinite(Cs) & np.isfinite(ys)
    Cs, ys = Cs[ok], ys[ok]
    if len(Cs) < 3:
        return float("nan")
    order = np.argsort(Cs)
    Cs, ys = Cs[order], ys[order]
    ys_s = ys.copy()
    ys_s[1:-1] = 0.25 * ys[:-2] + 0.5 * ys[1:-1] + 0.25 * ys[2:]
    slopes = np.diff(ys_s) / (np.diff(Cs) + 1e-12)
    if len(slopes) == 0:
        return float("nan")
    idx = int(np.argmax(slopes))
    return float(0.5 * (Cs[idx] + Cs[idx + 1]))


# -----------------------------
# Synthetic staged environment
# -----------------------------


@dataclass
class WorldSpec:
    obs_dim: int = 32
    spur_dim: int = 4
    n_actions: int = 4
    n_outcomes: int = 8
    horizon: int = 2
    noise_std: float = 0.15
    spur_strength: float = 1.0
    seed: int = 1234

    bit_emb: Optional[np.ndarray] = None
    upper_emb: Optional[np.ndarray] = None
    lower_emb: Optional[np.ndarray] = None
    inter_emb: Optional[np.ndarray] = None

    def build(self) -> "WorldSpec":
        rng = np.random.default_rng(self.seed)
        core_dim = self.obs_dim - self.spur_dim - 2
        assert core_dim >= 8, "obs_dim too small"
        self.bit_emb = rng.normal(0, 0.7, size=(6, core_dim)).astype(np.float32)
        self.upper_emb = rng.normal(0, 0.55, size=(8, core_dim)).astype(np.float32)
        self.lower_emb = rng.normal(0, 0.55, size=(8, core_dim)).astype(np.float32)
        self.inter_emb = rng.normal(0, 0.25, size=(8, 8, core_dim)).astype(np.float32)
        return self


def code_to_bits_np(code: np.ndarray) -> np.ndarray:
    return ((np.asarray(code, dtype=np.int64)[..., None] >> np.arange(5, -1, -1)) & 1).astype(np.float32)


class StagedDataset(Dataset):
    """Latent 64-state stage dataset with mixed train domains and OOD splits."""

    def __init__(self, n_seq: int, seq_len: int, world: WorldSpec, seed: int, split: str) -> None:
        assert split in {"train_mixed", "id", "ood_random", "ood_inverted"}
        self.n_seq = int(n_seq)
        self.seq_len = int(seq_len)
        self.world = world
        self.seed = int(seed)
        self.split = split
        self.x, self.action, self.outcome, self.code = self._generate()

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int):
        return self.x[idx], self.action[idx], self.outcome[idx], self.code[idx]

    def _transition_codes(self, rng: np.random.Generator, total_len: int) -> np.ndarray:
        upper = int(rng.integers(0, 8))
        lower = int(rng.integers(0, 8))
        out = []
        for _ in range(total_len):
            out.append(upper * 8 + lower)
            # Lower dynamics are fast and upper-gated.
            step = 1 + (upper % 3)
            if rng.random() < 0.78:
                lower = (lower + step + ((upper ^ lower) & 1)) % 8
            else:
                lower = int(rng.integers(0, 8))
            # Upper dynamics are slow and conflict-gated.
            conflict = ((upper ^ lower) & 1)
            if rng.random() < 0.10 + 0.18 * conflict:
                upper = upper ^ (1 << int(rng.integers(0, 3)))
        return np.asarray(out, dtype=np.int64)

    @staticmethod
    def _decision_target(cur_code: np.ndarray, fut_code: np.ndarray) -> np.ndarray:
        cur_u, cur_l = cur_code // 8, cur_code % 8
        fut_u, fut_l = fut_code // 8, fut_code % 8
        lower_delta = (fut_l - cur_l) % 8
        move_high = (lower_delta >= 3).astype(np.int64)
        upper_risk = (((fut_u >> 1) ^ (cur_u & 1)) & 1).astype(np.int64)
        return (2 * upper_risk + move_high).astype(np.int64)

    def _spurious_label(self, y: np.ndarray, upper: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.split == "id":
            return y.copy()
        if self.split == "ood_random":
            return rng.integers(0, self.world.spur_dim, size=y.shape).astype(np.int64)
        if self.split == "ood_inverted":
            return (y + 1 + (upper & 1)) % self.world.spur_dim
        # train_mixed: randomize spurious domain at episode level to prevent single shortcut.
        domain = int(rng.integers(0, 4))
        if domain == 0:
            return y.copy()
        if domain == 1:
            return (y + 1) % self.world.spur_dim
        if domain == 2:
            return rng.integers(0, self.world.spur_dim, size=y.shape).astype(np.int64)
        return (y + 2 + (upper & 1)) % self.world.spur_dim

    def _generate(self):
        rng = np.random.default_rng(self.seed)
        T, H = self.seq_len, self.world.horizon
        obs = np.zeros((self.n_seq, T, self.world.obs_dim), dtype=np.float32)
        actions = np.zeros((self.n_seq, T), dtype=np.int64)
        outcomes = np.zeros((self.n_seq, T), dtype=np.int64)
        codes_out = np.zeros((self.n_seq, T), dtype=np.int64)
        core_dim = self.world.obs_dim - self.world.spur_dim - 2

        assert self.world.bit_emb is not None
        assert self.world.upper_emb is not None
        assert self.world.lower_emb is not None
        assert self.world.inter_emb is not None

        for n in range(self.n_seq):
            codes = self._transition_codes(rng, T + H + 1)
            cur, fut = codes[:T], codes[H:H + T]
            upper, lower = cur // 8, cur % 8
            fut_upper = fut // 8
            y = self._decision_target(cur, fut)
            bits = code_to_bits_np(cur)
            core = bits @ self.world.bit_emb
            core += self.world.upper_emb[upper]
            core += self.world.lower_emb[lower]
            core += 0.65 * self.world.inter_emb[upper, lower]
            core += rng.normal(0, self.world.noise_std, size=core.shape).astype(np.float32)

            phase = np.arange(T, dtype=np.float32) / max(T - 1, 1)
            time_feats = np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=-1).astype(np.float32)
            spur_label = self._spurious_label(y, upper, rng)
            spur = np.eye(self.world.spur_dim, dtype=np.float32)[spur_label % self.world.spur_dim]
            spur = self.world.spur_strength * spur
            spur += rng.normal(0, 0.25, size=spur.shape).astype(np.float32)
            obs[n] = np.concatenate([core[:, :core_dim], spur, time_feats], axis=-1)
            actions[n] = y
            outcomes[n] = fut_upper
            codes_out[n] = cur

        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(outcomes, dtype=torch.long),
            torch.tensor(codes_out, dtype=torch.long),
        )


# -----------------------------
# Model
# -----------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SymbolicBottleneckModel(nn.Module):
    """Minimal discrete model with C-controlled symbolic coupling.

    C controls gate = C / (C + c_scale).

    residual-strength mode:
      q = gate(C) * symbolic_embedding + residual_strength * (1-gate(C)) * continuous_projection(z)

    residual_strength=0.0 is pure symbolic, with no continuous leakage.
    residual_strength=1.0 is the full hybrid path.

    original:    logits = linear(z)
    pre_softmax: logits = linear(tanh(1.5 z)), i.e. nonlinearity before categorical logits.
    """

    def __init__(
        self,
        obs_dim: int,
        K: int,
        n_actions: int,
        n_outcomes: int,
        variant: str = "original",
        hidden_dim: int = 96,
        embed_dim: int = 16,
        c_scale: float = 2.0,
        mode: str = "residual_bypass",
        residual_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        assert variant in {"original", "pre_softmax"}
        assert mode in {"residual_bypass", "continuous_only"}
        self.K = int(K)
        self.variant = variant
        self.mode = mode
        # Kept as residual_lambda internally for compatibility with older code;
        # it is written to CSV as residual_strength.
        self.residual_lambda = float(residual_lambda)
        self.embed_dim = embed_dim
        self.c_scale = c_scale
        self.encoder = MLP(obs_dim, hidden_dim, embed_dim, n_layers=3)
        self.cont_proj = nn.Linear(embed_dim, embed_dim)
        self.logit_head = nn.Linear(embed_dim, K)
        self.codebook = nn.Parameter(torch.randn(K, embed_dim) / math.sqrt(embed_dim))
        self.action_head = MLP(embed_dim, hidden_dim, n_actions, n_layers=2)
        self.outcome_head = MLP(embed_dim, hidden_dim, n_outcomes, n_layers=2)
        self.trans_head = MLP(embed_dim, hidden_dim, K, n_layers=2)

    def gate(self, C: float) -> float:
        C = max(0.0, float(C))
        return C / (C + self.c_scale) if C > 0 else 0.0

    def forward(self, x: torch.Tensor, C: float, tau: float = 0.7, hard: bool = False) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        z_logits = torch.tanh(1.5 * z) if self.variant == "pre_softmax" else z
        logits = self.logit_head(z_logits)
        if hard:
            ids = torch.argmax(logits, dim=-1)
            probs = F.one_hot(ids, self.K).float()
        else:
            probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            ids = torch.argmax(probs, dim=-1)
        sym = probs @ self.codebook
        cont = self.cont_proj(z)
        g = self.gate(C)
        if self.mode == "continuous_only":
            q = cont
            residual_path = cont
            symbol_path = torch.zeros_like(sym)
        else:
            # D4b residual bypass schedule.
            # residual_strength=0: pure symbolic, no continuous leakage.
            # residual_strength=1: full hybrid, q = g*sym + (1-g)*cont.
            symbol_path = g * sym
            residual_path = self.residual_lambda * (1.0 - g) * cont
            q = symbol_path + residual_path
        # Diagnostics for whether nominal residual_strength matches actual bypass strength.
        with torch.no_grad():
            sym_var = symbol_path.detach().float().var(dim=0, unbiased=False).mean()
            res_var = residual_path.detach().float().var(dim=0, unbiased=False).mean()
            residual_var_ratio = res_var / (sym_var + 1e-8)
            sym_norm = symbol_path.detach().float().norm(dim=-1).mean()
            res_norm = residual_path.detach().float().norm(dim=-1).mean()
            residual_norm_ratio = res_norm / (sym_norm + 1e-8)
        return {
            "z": z,
            "q": q,
            "residual_var_ratio": residual_var_ratio,
            "residual_norm_ratio": residual_norm_ratio,
            "logits": logits,
            "probs": probs,
            "ids": ids,
            "action_logits": self.action_head(q),
            "outcome_logits": self.outcome_head(q),
            "trans_logits": self.trans_head(q),
        }


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    outcome_coef: float = 0.50
    transition_coef_base: float = 0.20
    usage_coef_base: float = 0.06
    cond_entropy_coef: float = 0.01
    grad_clip: float = 1.0
    tau_start: float = 1.2
    tau_min: float = 0.35


def tau_schedule(step: int, steps: int, start: float, minval: float) -> float:
    p = step / max(steps - 1, 1)
    return float(max(minval, start * ((minval / start) ** p)))


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train_one(
    model: SymbolicBottleneckModel,
    train_ds: Dataset,
    C: float,
    cfg: TrainConfig,
    device: torch.device,
    verbose: bool = False,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    scaler_enabled: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> None:
    model.to(device)
    model.train()
    loader = make_loader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    it = infinite_loader(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and scaler_enabled))
    g = model.gate(C)
    logK = math.log(model.K)
    for step in range(cfg.steps):
        x, y, outcome, code = next(it)
        x, y, outcome = to_device(x, device), to_device(y, device), to_device(outcome, device)
        B, T, D = x.shape
        tau = tau_schedule(step, cfg.steps, cfg.tau_start, cfg.tau_min)
        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(x.reshape(B * T, D), C=C, tau=tau, hard=False)
            a_logits = out["action_logits"].reshape(B, T, -1)
            o_logits = out["outcome_logits"].reshape(B, T, -1)
            loss = F.cross_entropy(a_logits.reshape(B * T, -1), y.reshape(-1))
            loss = loss + cfg.outcome_coef * F.cross_entropy(o_logits.reshape(B * T, -1), outcome.reshape(-1))

            probs = out["probs"]
            ids = out["ids"].reshape(B, T)
            pbar = probs.mean(0)
            batch_ent = -(pbar * torch.log(pbar + 1e-9)).sum()
            cond_ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
            # Encourage clear per-sample codes and non-collapsed global usage, more strongly as C grows.
            loss = loss + cfg.cond_entropy_coef * cond_ent / logK
            loss = loss - cfg.usage_coef_base * g * batch_ent / logK

            if T > 1 and g > 0:
                trans_logits = out["trans_logits"].reshape(B, T, model.K)
                loss_trans = F.cross_entropy(trans_logits[:, :-1, :].reshape(-1, model.K), ids[:, 1:].reshape(-1).detach())
                loss = loss + cfg.transition_coef_base * g * loss_trans

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        if verbose and (step % max(1, cfg.steps // 4) == 0 or step == cfg.steps - 1):
            acc = (a_logits.argmax(-1) == y).float().mean().item()
            ppl = torch.exp(batch_ent.float()).item()
            print(f"    step={step:04d} loss={loss.item():.3f} acc={acc:.3f} tau={tau:.2f} gate={g:.2f} ppl={ppl:.1f}/{model.K}", flush=True)


@torch.no_grad()
def evaluate(
    model: SymbolicBottleneckModel,
    ds: StagedDataset,
    C: float,
    device: torch.device,
    batch_size: int = 256,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> Dict[str, float]:
    model.eval()
    loader = make_loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    total = 0
    correct_action = 0
    correct_outcome = 0
    transition_head_correct = 0
    transition_head_total = 0
    all_q, all_ids, all_y, all_outcome, all_code = [], [], [], [], []
    residual_var_ratios: List[float] = []
    residual_norm_ratios: List[float] = []
    for x, y, outcome, code in loader:
        x_dev = to_device(x, device)
        B, T, D = x_dev.shape
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(x_dev.reshape(B * T, D), C=C, tau=0.1, hard=True)
            if "residual_var_ratio" in out:
                residual_var_ratios.append(float(out["residual_var_ratio"].detach().cpu().item()))
                residual_norm_ratios.append(float(out["residual_norm_ratio"].detach().cpu().item()))
            a_logits = out["action_logits"].reshape(B, T, -1)
            o_logits = out["outcome_logits"].reshape(B, T, -1)
            trans_logits = out["trans_logits"].reshape(B, T, -1)
            trans_pred = trans_logits.argmax(-1)
            ids_batch = out["ids"].reshape(B, T)
            if T > 1:
                transition_head_correct += int((trans_pred[:, :-1] == ids_batch[:, 1:]).sum().item())
                transition_head_total += B * (T - 1)
        a_logits_cpu = a_logits.float().cpu()
        o_logits_cpu = o_logits.float().cpu()
        pred_a = a_logits_cpu.argmax(-1)
        pred_o = o_logits_cpu.argmax(-1)
        total += B * T
        correct_action += int((pred_a == y).sum().item())
        correct_outcome += int((pred_o == outcome).sum().item())
        all_q.append(out["q"].float().cpu().numpy())
        all_ids.append(out["ids"].cpu().numpy())
        all_y.append(y.numpy().reshape(-1))
        all_outcome.append(outcome.numpy().reshape(-1))
        all_code.append(code.numpy().reshape(-1))

    q = np.concatenate(all_q, axis=0)
    ids = np.concatenate(all_ids, axis=0).astype(np.int64)
    y = np.concatenate(all_y).astype(np.int64)
    outcome = np.concatenate(all_outcome).astype(np.int64)
    true_code = np.concatenate(all_code).astype(np.int64)
    upper = true_code // 8
    lower = true_code % 8

    action_acc = correct_action / max(total, 1)
    outcome_acc = correct_outcome / max(total, 1)
    transition_head_acc = transition_head_correct / max(transition_head_total, 1)
    majority_acc = float(np.max(np.bincount(y, minlength=4)) / len(y))
    utility = (action_acc - majority_acc) / max(1e-9, 1.0 - majority_acc)

    K = model.K
    counts = np.bincount(ids, minlength=K).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    ppl = float(np.exp(entropy_np(p)))

    def contingency(labels: np.ndarray, n_label: int) -> np.ndarray:
        m = np.zeros((K, n_label), dtype=np.float64)
        np.add.at(m, (ids, labels), 1)
        return m

    nmi_action = normalized_mi(contingency(y, 4))
    nmi_outcome = normalized_mi(contingency(outcome, 8))
    nmi_upper = normalized_mi(contingency(upper, 8))
    nmi_lower = normalized_mi(contingency(lower, 8))

    # Transition predictability from hard symbol sequence.
    ids_seq = ids.reshape(ds.n_seq, ds.seq_len)
    src = ids_seq[:, :-1].reshape(-1)
    dst = ids_seq[:, 1:].reshape(-1)
    trans = np.zeros((K, K), dtype=np.float64)
    np.add.at(trans, (src, dst), 1)
    trans_total = trans.sum()
    if trans_total > 0:
        trans_acc = float(np.max(trans, axis=1).sum() / trans_total)
        next_counts = trans.sum(axis=0)
        trans_majority = float(np.max(next_counts) / trans_total)
        trans_acc_gain = (trans_acc - trans_majority) / max(1e-9, 1.0 - trans_majority)
    else:
        trans_acc = trans_majority = trans_acc_gain = float("nan")

    # Unlike transition_acc_gain above, this readout uses the model's
    # transition head on q(C), so it can vary with C after weights are frozen.
    head_next_counts = np.bincount(ids_seq[:, 1:].reshape(-1), minlength=K)
    head_majority = float(np.max(head_next_counts) / max(transition_head_total, 1))
    transition_head_gain = (transition_head_acc - head_majority) / max(1e-9, 1.0 - head_majority)

    er = effective_rank(q)
    er_norm = er / max(1.0, q.shape[1])

    # Dynamics score: normalized later across C. Store raw components here.
    return {
        "action_acc": action_acc,
        "outcome_acc": outcome_acc,
        "majority_acc": majority_acc,
        "utility": utility,
        "erank": er,
        "erank_norm": er_norm,
        "code_ppl": ppl,
        "code_ppl_frac": ppl / K,
        "nmi_action": nmi_action,
        "nmi_outcome": nmi_outcome,
        "nmi_upper": nmi_upper,
        "nmi_lower": nmi_lower,
        "transition_acc": trans_acc,
        "transition_majority": trans_majority,
        "transition_acc_gain": trans_acc_gain,
        "transition_head_acc": transition_head_acc,
        "transition_head_majority": head_majority,
        "transition_head_gain": transition_head_gain,
    }


# -----------------------------
# C* decomposition
# -----------------------------


def add_dyn_scores(curves: pd.DataFrame) -> pd.DataFrame:
    """Add per-curve normalized rank/dynamics scores.

    dyn_score_no_trans is the recommended default after the Markov-baseline finding:
      D_no_trans = 0.60 * nmi_outcome + 0.40 * nmi_lower

    dyn_score_simple keeps transition_acc_gain for diagnostics only:
      D_simple = 0.40 * nmi_outcome + 0.30 * nmi_lower + 0.30 * transition_acc_gain
    """
    curves = curves.copy()
    curves["rank_score"] = np.nan
    curves["dyn_score_no_trans"] = np.nan
    curves["dyn_score_simple"] = np.nan
    curves["dyn_score"] = np.nan
    group_cols = ["mode", "residual_lambda", "variant", "seed", "K", "split"]
    for key, g in curves.groupby(group_cols, sort=False):
        idx = g.index
        R = normalize_curve_values(g["erank_norm"].values)
        nmi_o = normalize_curve_values(g["nmi_outcome"].values)
        nmi_l = normalize_curve_values(g["nmi_lower"].values)
        trans = normalize_curve_values(g["transition_acc_gain"].values)
        D_no_trans = 0.60 * nmi_o + 0.40 * nmi_l
        D_simple = 0.40 * nmi_o + 0.30 * nmi_l + 0.30 * trans
        curves.loc[idx, "rank_score"] = R
        curves.loc[idx, "dyn_score_no_trans"] = D_no_trans
        curves.loc[idx, "dyn_score_simple"] = D_simple
        curves.loc[idx, "dyn_score"] = D_no_trans
    return curves


def extract_cstars(
    curves: pd.DataFrame,
    frac: float = 0.90,
    min_gain: float = 0.10,
    rank_min_gain: float = 0.05,
    dyn_min_gain: float = 0.05,
) -> pd.DataFrame:
    rows = []
    group_cols = ["mode", "residual_lambda", "variant", "seed", "K", "split"]
    for key, g in curves.groupby(group_cols, sort=False):
        # Pull identifiers from the group itself rather than unpacking the groupby key.
        # This is robust to pandas versions / duplicate-label edge cases.
        mode = str(g["mode"].iloc[0])
        residual_lambda = float(g["residual_lambda"].iloc[0])
        variant = str(g["variant"].iloc[0])
        seed = int(g["seed"].iloc[0])
        K = int(g["K"].iloc[0])
        split = str(g["split"].iloc[0])
        g = g.sort_values("C")
        Cs = g["C"].values
        R = g["rank_score"].values
        D = g["dyn_score"].values
        U = g["utility"].values
        rank_gain = curve_gain(R)
        dyn_gain = curve_gain(D)
        util_gain = curve_gain(U)
        utility_sensitivity = float(np.nanmax(U) - np.nanmin(U)) if np.isfinite(U).any() else float("nan")
        C_rank_raw = critical_C_threshold(Cs, R, frac=frac, min_gain=0.0)
        C_dyn_raw = critical_C_threshold(Cs, D, frac=frac, min_gain=0.0)
        C_rank = critical_C_threshold(Cs, R, frac=frac, min_gain=rank_min_gain)
        C_dyn = critical_C_threshold(Cs, D, frac=frac, min_gain=dyn_min_gain)
        C_util = critical_C_threshold(Cs, U, frac=frac, min_gain=min_gain)
        C_rank_slope = critical_C_slope(Cs, R)
        C_dyn_slope = critical_C_slope(Cs, D)
        C_util_slope = critical_C_slope(Cs, U)
        residual_var_ratio = float(np.nanmean(g["residual_var_ratio"].values)) if "residual_var_ratio" in g.columns else float("nan")
        residual_norm_ratio = float(np.nanmean(g["residual_norm_ratio"].values)) if "residual_norm_ratio" in g.columns else float("nan")
        E_rank = abs(C_rank - C_util) if np.isfinite(C_rank) and np.isfinite(C_util) else float("nan")
        E_dyn = abs(C_dyn - C_util) if np.isfinite(C_dyn) and np.isfinite(C_util) else float("nan")
        rows.append({
            "mode": mode,
            "residual_lambda": float(residual_lambda),
            "variant": variant,
            "residual_strength": float(residual_lambda),
            "seed": int(seed),
            "K": int(K),
            "split": split,
            "C_rank": C_rank,
            "C_dyn": C_dyn,
            "C_util": C_util,
            "C_rank_raw": C_rank_raw,
            "C_dyn_raw": C_dyn_raw,
            "rank_gain": rank_gain,
            "dyn_gain": dyn_gain,
            "util_gain": util_gain,
            "rank_valid": float(rank_gain >= rank_min_gain),
            "dyn_valid": float(dyn_gain >= dyn_min_gain),
            "utility_sensitive": float(util_gain >= min_gain),
            "utility_sensitivity": utility_sensitivity,
            "residual_var_ratio": residual_var_ratio,
            "residual_norm_ratio": residual_norm_ratio,
            "Delta_C_dyn_eq": (C_dyn - C_rank) if np.isfinite(C_dyn) and np.isfinite(C_rank) else float("nan"),
            "sync_gap": abs(C_dyn - C_rank) if np.isfinite(C_dyn) and np.isfinite(C_rank) else float("nan"),
            "dyn_after_eq": float(C_dyn > C_rank) if np.isfinite(C_dyn) and np.isfinite(C_rank) else float("nan"),
            "rank_util_gap": E_rank,
            "dyn_util_gap": E_dyn,
            "E_rank": E_rank,
            "E_dyn": E_dyn,
            "dyn_better": float(E_dyn < E_rank) if np.isfinite(E_rank) and np.isfinite(E_dyn) else float("nan"),
        })
    return pd.DataFrame(rows)



def confirmation_summary(summary: pd.DataFrame, sync_tol: float = 1.0, early_c: float = 2.0, late_c: float = 8.0) -> pd.DataFrame:
    """Summarize D4-v4 channel-ordering diagnostics.

    Delta is defined as C_dyn - C_rank.
      Delta < 0: dynamics-level C* occurs earlier than rank/equivalence C*.
      Delta > 0: dynamics-level C* occurs later than rank/equivalence C*.
      |Delta| <= sync_tol: synchronized emergence.

    Main confirmation fields:
      sync_rate:          mean(|Delta| <= sync_tol)
      negative_delta_rate: mean(Delta < 0)
      positive_delta_rate: mean(Delta > 0)
      early_dyn_rate:     mean(C_dyn <= early_c)
      late_rank_rate:     mean(C_rank >= late_c)
      abs_delta_mean:     mean(|Delta|)
    """
    if len(summary) == 0:
        return pd.DataFrame()
    df = summary.copy()
    delta = df["Delta_C_dyn_eq"]
    df["abs_delta"] = delta.abs()
    df["sync"] = (df["abs_delta"] <= sync_tol).astype(float)
    df["negative_delta"] = (delta < 0).astype(float)
    df["positive_delta"] = (delta > 0).astype(float)
    df["early_dyn"] = (df["C_dyn"] <= early_c).astype(float)
    df["late_rank"] = (df["C_rank"] >= late_c).astype(float)
    df["both_early_dyn_late_rank"] = ((df["C_dyn"] <= early_c) & (df["C_rank"] >= late_c)).astype(float)

    group_cols = ["mode", "residual_strength", "variant", "K", "split"]
    out = df.groupby(group_cols).agg(
        n=("Delta_C_dyn_eq", "count"),
        C_rank_mean=("C_rank", "mean"),
        C_rank_std=("C_rank", "std"),
        C_dyn_mean=("C_dyn", "mean"),
        C_dyn_std=("C_dyn", "std"),
        delta_mean=("Delta_C_dyn_eq", "mean"),
        delta_std=("Delta_C_dyn_eq", "std"),
        abs_delta_mean=("abs_delta", "mean"),
        abs_delta_std=("abs_delta", "std"),
        sync_rate=("sync", "mean"),
        negative_delta_rate=("negative_delta", "mean"),
        positive_delta_rate=("positive_delta", "mean"),
        early_dyn_rate=("early_dyn", "mean"),
        late_rank_rate=("late_rank", "mean"),
        early_dyn_late_rank_rate=("both_early_dyn_late_rank", "mean"),
        rank_gain_mean=("rank_gain", "mean"),
        dyn_gain_mean=("dyn_gain", "mean"),
        util_gain_mean=("util_gain", "mean"),
        rank_valid_rate=("rank_valid", "mean"),
        dyn_valid_rate=("dyn_valid", "mean"),
        utility_sensitive_rate=("utility_sensitive", "mean"),
    ).reset_index()
    return out


def mode_level_confirmation(summary: pd.DataFrame, sync_tol: float = 1.0, early_c: float = 2.0, late_c: float = 8.0) -> pd.DataFrame:
    """Coarser summary across K/split for quick yes/no checks."""
    if len(summary) == 0:
        return pd.DataFrame()
    df = summary.copy()
    delta = df["Delta_C_dyn_eq"]
    df["abs_delta"] = delta.abs()
    df["sync"] = (df["abs_delta"] <= sync_tol).astype(float)
    df["negative_delta"] = (delta < 0).astype(float)
    df["positive_delta"] = (delta > 0).astype(float)
    df["early_dyn"] = (df["C_dyn"] <= early_c).astype(float)
    df["late_rank"] = (df["C_rank"] >= late_c).astype(float)
    df["both_early_dyn_late_rank"] = ((df["C_dyn"] <= early_c) & (df["C_rank"] >= late_c)).astype(float)
    out = df.groupby(["mode", "residual_strength", "variant"]).agg(
        n=("Delta_C_dyn_eq", "count"),
        delta_mean=("Delta_C_dyn_eq", "mean"),
        abs_delta_mean=("abs_delta", "mean"),
        sync_rate=("sync", "mean"),
        negative_delta_rate=("negative_delta", "mean"),
        positive_delta_rate=("positive_delta", "mean"),
        early_dyn_rate=("early_dyn", "mean"),
        late_rank_rate=("late_rank", "mean"),
        early_dyn_late_rank_rate=("both_early_dyn_late_rank", "mean"),
        rank_valid_rate=("rank_valid", "mean"),
        dyn_valid_rate=("dyn_valid", "mean"),
        utility_sensitive_rate=("utility_sensitive", "mean"),
    ).reset_index()
    return out

def make_plots(curves: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> None:
    if not HAS_MPL:
        return
    os.makedirs(out_dir, exist_ok=True)
    # Mean curves by variant/K/split.
    for split in curves["split"].unique():
        sub = curves[curves["split"] == split]
        for K in sorted(sub["K"].unique()):
            plt.figure(figsize=(9, 5))
            for mode in sub["mode"].unique():
                for residual_lambda in sorted(sub[sub["mode"] == mode]["residual_lambda"].unique()):
                    for variant in sub["variant"].unique():
                        g = sub[(sub["mode"] == mode) & (sub["residual_lambda"] == residual_lambda) & (sub["variant"] == variant) & (sub["K"] == K)]
                        if len(g) == 0:
                            continue
                        mode_label = f"{mode}(λ={residual_lambda:g})" if mode == "symbol_residual" else mode
                        m = g.groupby("C")[["rank_score", "dyn_score", "utility"]].mean().reset_index()
                        label = f"{mode_label}/{variant}"
                        plt.plot(m["C"], m["rank_score"], marker="o", label=f"{label}: rank")
                        plt.plot(m["C"], m["dyn_score"], marker="o", label=f"{label}: dyn")
                        plt.plot(m["C"], m["utility"], marker="o", label=f"{label}: util")
            plt.title(f"C curves, split={split}, K={K}")
            plt.xlabel("C")
            plt.ylabel("score")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"curves_{split}_K{K}.png"), dpi=160)
            plt.close()

    if len(summary) > 0:
        agg = summary.groupby(["mode", "residual_strength", "variant", "K", "split"])[["E_rank", "E_dyn", "dyn_better", "Delta_C_dyn_eq"]].mean().reset_index()
        for split in agg["split"].unique():
            sub = agg[agg["split"] == split].copy()
            labels = [f"{r.mode}/{r.variant}\nK={int(r.K)}" for r in sub.itertuples()]
            x = np.arange(len(sub))
            width = 0.38
            plt.figure(figsize=(max(8, len(sub) * 0.8), 4.5))
            plt.bar(x - width / 2, sub["E_rank"], width, label="|C_rank-C_util|")
            plt.bar(x + width / 2, sub["E_dyn"], width, label="|C_dyn-C_util|")
            plt.xticks(x, labels, rotation=30, ha="right")
            plt.ylabel("C* error")
            plt.title(f"C* prediction errors, split={split}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"cstar_errors_{split}.png"), dpi=160)
            plt.close()


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="runs/d4b_residual_desync")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", choices=["off", "auto", "fp16", "bf16"], default="auto",
                   help="CUDA mixed precision. auto=bf16 on Ampere+ else fp16; off disables AMP.")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on Ampere+ GPUs.")
    p.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high",
                   help="Passed to torch.set_float32_matmul_precision when available.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers. Use 2-4 on GPU; 0 if it causes issues.")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs/loaders.")
    p.add_argument("--eval-batch-size", type=int, default=0, help="0 means max(batch_size,1024) on CUDA, max(batch_size,256) on CPU.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile(model) when available. Good for longer GPU runs, not smoke tests.")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--Ks", nargs="+", type=int, default=[8, 16])
    p.add_argument("--Cs", nargs="+", type=float, default=[0, 0.5, 1, 2, 3, 5, 8, 12, 16, 24])
    p.add_argument("--variants", nargs="+", choices=["original", "pre_softmax"], default=["original"])
    p.add_argument("--residual-strengths", nargs="+", type=float, default=[0.0, 0.10, 0.25, 0.50, 1.0],
                   help="Residual bypass strengths. 0=pure symbolic; 1=full hybrid; q=g*symbol + strength*(1-g)*continuous.")
    p.add_argument("--include-continuous-only", action="store_true",
                   help="Also run a continuous_only diagnostic baseline. It is excluded by default.")
    p.add_argument("--splits", nargs="+", choices=["id", "ood_random", "ood_inverted"], default=["id", "ood_random", "ood_inverted"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-train", type=int, default=2048)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--frac", type=float, default=0.90)
    p.add_argument("--min-gain", type=float, default=0.10, help="Minimum utility gain required to define C*_util")
    p.add_argument("--rank-min-gain", type=float, default=0.05, help="Minimum rank-score gain required to define gated C*_rank")
    p.add_argument("--dyn-min-gain", type=float, default=0.05, help="Minimum dynamics-score gain required to define gated C*_dyn")
    p.add_argument("--c-scale", type=float, default=2.0)
    p.add_argument("--world-seed", type=int, default=1234)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--torch-threads", type=int, default=0, help="CPU torch threads. 0 leaves PyTorch default; only relevant on CPU/data loading.")
    p.add_argument("--sync-tol", type=float, default=1.0, help="|Delta| <= sync_tol counts as synchronized C* emergence.")
    p.add_argument("--early-c", type=float, default=2.0, help="C_dyn <= early_c counts as early dynamics C*.")
    p.add_argument("--late-c", type=float, default=8.0, help="C_rank >= late_c counts as late rank/equivalence C*.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.torch_threads is not None and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    configure_gpu(device, tf32=args.tf32, matmul_precision=args.matmul_precision)
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(device, args.amp)
    if args.eval_batch_size and args.eval_batch_size > 0:
        eval_batch_size = int(args.eval_batch_size)
    else:
        eval_batch_size = max(args.batch_size, 1024 if device.type == "cuda" else 256)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"AMP: enabled={amp_enabled}, dtype={amp_dtype}, scaler={scaler_enabled}; TF32={args.tf32}")
    print(f"DataLoader: workers={args.num_workers}, pin_memory={device.type == 'cuda'}, eval_batch_size={eval_batch_size}")
    print(f"Output dir: {args.out_dir}")

    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "train_config": asdict(train_cfg)}, f, indent=2, ensure_ascii=False)

    world = WorldSpec(obs_dim=args.obs_dim, seed=args.world_seed).build()
    rows: List[Dict[str, float]] = []
    t0_all = time.time()

    for seed in args.seeds:
        train_ds = StagedDataset(args.n_train, args.seq_len, world, seed=10000 + seed, split="train_mixed")
        eval_sets = {
            split: StagedDataset(args.n_test, args.seq_len, world, seed=20000 + 17 * seed + i, split=split)
            for i, split in enumerate(args.splits)
        }
        run_conditions = [("residual_bypass", float(s)) for s in args.residual_strengths]
        if args.include_continuous_only:
            run_conditions.append(("continuous_only", 0.0))
        for mode, residual_lambda in run_conditions:
            for variant in args.variants:
                for K in args.Ks:
                    for C in args.Cs:
                        label = f"residual_strength={residual_lambda:g}" if mode == "residual_bypass" else "continuous_only"
                        print(f"\n=== seed={seed} {label} variant={variant} K={K} C={C} ===", flush=True)
                        seed_offset = int(1000 * residual_lambda) + (101 if mode == "continuous_only" else 0)
                        set_seed(seed + 1000 * K + int(100 * C) + (0 if variant == "original" else 7) + seed_offset)
                        model = SymbolicBottleneckModel(
                            obs_dim=args.obs_dim,
                            K=K,
                            n_actions=world.n_actions,
                            n_outcomes=world.n_outcomes,
                            variant=variant,
                            hidden_dim=args.hidden_dim,
                            embed_dim=args.embed_dim,
                            c_scale=args.c_scale,
                            mode=mode,
                            residual_lambda=residual_lambda,
                        )
                        if args.compile:
                            try:
                                model = torch.compile(model)  # type: ignore[attr-defined]
                            except Exception as e:
                                print(f"[compile warning] torch.compile failed, using eager mode: {e}", flush=True)
                        train_one(
                            model, train_ds, C=C, cfg=train_cfg, device=device, verbose=args.verbose,
                            amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler_enabled=scaler_enabled,
                            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                            persistent_workers=args.persistent_workers,
                        )
                        for split, ds in eval_sets.items():
                            metrics = evaluate(
                                model, ds, C=C, device=device, batch_size=eval_batch_size,
                                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                                num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                                persistent_workers=args.persistent_workers,
                            )
                            metrics.update({"seed": seed, "mode": mode, "residual_lambda": residual_lambda, "residual_strength": residual_lambda, "variant": variant, "K": K, "C": C, "split": split})
                            rows.append(metrics)
                            print(
                                f"  {split}: U={metrics['utility']:.3f} acc={metrics['action_acc']:.3f} "
                                f"R={metrics['erank_norm']:.3f} nmi_out={metrics['nmi_outcome']:.3f} "
                                f"nmi_low={metrics['nmi_lower']:.3f} trans={metrics['transition_acc_gain']:.3f}",
                                flush=True,
                            )

    curves = pd.DataFrame(rows)
    if "residual_strength" not in curves.columns and "residual_lambda" in curves.columns:
        curves["residual_strength"] = curves["residual_lambda"]
    curves = add_dyn_scores(curves)
    summary = extract_cstars(curves, frac=args.frac, min_gain=args.min_gain, rank_min_gain=args.rank_min_gain, dyn_min_gain=args.dyn_min_gain)
    if "residual_strength" not in summary.columns and "residual_lambda" in summary.columns:
        summary["residual_strength"] = summary["residual_lambda"]
    aggregate = summary.groupby(["mode", "residual_strength", "variant", "K", "split"])[[
        "E_rank", "E_dyn", "dyn_better", "C_rank", "C_dyn", "C_util",
        "C_rank_raw", "C_dyn_raw", "rank_gain", "dyn_gain", "util_gain",
        "rank_valid", "dyn_valid", "utility_sensitive", "utility_sensitivity", "Delta_C_dyn_eq", "sync_gap", "rank_util_gap", "dyn_util_gap", "dyn_after_eq"
    ]].agg(["mean", "std"]).reset_index()
    confirm = confirmation_summary(summary, sync_tol=args.sync_tol, early_c=args.early_c, late_c=args.late_c)
    confirm_mode = mode_level_confirmation(summary, sync_tol=args.sync_tol, early_c=args.early_c, late_c=args.late_c)
    # D4b-specific summaries: keep ID/OOD split separate.
    d4b_by_strength = summary.copy()
    d4b_by_strength["negative_delta"] = (d4b_by_strength["Delta_C_dyn_eq"] < 0).astype(float)
    d4b_by_strength["positive_delta"] = (d4b_by_strength["Delta_C_dyn_eq"] > 0).astype(float)
    d4b_by_strength["sync"] = (d4b_by_strength["sync_gap"] <= args.sync_tol).astype(float)
    d4b_by_strength = d4b_by_strength.groupby(["residual_strength", "variant", "K", "split"]).agg(
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
        residual_var_ratio_mean=("residual_var_ratio", "mean") if "residual_var_ratio" in summary.columns else ("sync_gap", "mean"),
        residual_norm_ratio_mean=("residual_norm_ratio", "mean") if "residual_norm_ratio" in summary.columns else ("sync_gap", "mean"),
    ).reset_index()
    d4b_strength_summary = d4b_by_strength.groupby(["residual_strength", "variant", "split"]).agg(
        n=("n", "sum"),
        sync_gap_mean=("sync_gap_mean", "mean"),
        sync_rate_mean=("sync_rate", "mean"),
        utility_sensitivity_mean=("utility_sensitivity_mean", "mean"),
        negative_delta_rate_mean=("negative_delta_rate", "mean"),
        positive_delta_rate_mean=("positive_delta_rate", "mean"),
        dyn_util_gap_mean=("dyn_util_gap_mean", "mean"),
        rank_util_gap_mean=("rank_util_gap_mean", "mean"),
    ).reset_index()

    curves_path = os.path.join(args.out_dir, "curves.csv")
    summary_path = os.path.join(args.out_dir, "cstar_summary.csv")
    aggregate_path = os.path.join(args.out_dir, "aggregate.csv")
    confirm_path = os.path.join(args.out_dir, "confirmation_summary.csv")
    confirm_mode_path = os.path.join(args.out_dir, "confirmation_by_mode.csv")
    d4b_by_strength_path = os.path.join(args.out_dir, "d4b_by_strength.csv")
    d4b_strength_summary_path = os.path.join(args.out_dir, "d4b_strength_summary.csv")
    curves.to_csv(curves_path, index=False)
    summary.to_csv(summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    confirm.to_csv(confirm_path, index=False)
    confirm_mode.to_csv(confirm_mode_path, index=False)
    d4b_by_strength.to_csv(d4b_by_strength_path, index=False)
    d4b_strength_summary.to_csv(d4b_strength_summary_path, index=False)

    if not args.no_plots:
        make_plots(curves, summary, args.out_dir)

    print("\n===== C* decomposition summary =====")
    printable = summary.groupby(["mode", "residual_strength", "variant", "K", "split"])[[
        "C_rank", "C_dyn", "Delta_C_dyn_eq", "rank_gain", "rank_valid",
        "E_rank", "E_dyn", "dyn_better", "util_gain", "utility_sensitive"
    ]].mean().reset_index()
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(printable)
    print("\n===== D4 confirmation by mode =====")
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(confirm_mode)
    print("\nInterpretation:")
    print("  Delta_C_dyn_eq = C_dyn - C_rank.")
    print("    Delta < 0: dynamics-level C* occurs earlier than rank/equivalence C*.")
    print("    Delta > 0: dynamics-level C* occurs later than rank/equivalence C*.")
    print("    |Delta| <= sync_tol counts as synchronized emergence.")
    print("  H4a: symbol_only should have high sync_rate.")
    print("  H4b: symbol_residual should show larger abs_delta as residual_lambda grows, especially in OOD.")
    print("  H4c: hybrid should show high negative_delta_rate and large abs_delta if continuous bypass induces early-dynamics / late-equivalence separation.")
    print(f"\nSaved:\n  {curves_path}\n  {summary_path}\n  {aggregate_path}\n  {confirm_path}\n  {confirm_mode_path}")
    if not args.no_plots and HAS_MPL:
        print(f"  plots in {args.out_dir}/*.png")
    print(f"Total time: {time.time() - t0_all:.1f}s")


if __name__ == "__main__":
    main()
