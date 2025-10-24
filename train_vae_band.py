#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae_band.py

VAE that learns a diagonal preconditioner M for SPD matrices (S^2 x S^2),
using a contiguous diagonal band input of width W (offsets -W..+W).

- Input per sample: X in R^{(2W+1) x S^2}, normalized by a per-sample scale.
- Decoder predicts s = log(m/diag(A)) and outputs m_diag (length S^2) via:
    m = diag(A) * softplus(s) + eps  > 0
- Loss = residual proxy (low-variance Hutchinson) + spectral spread proxy
         + beta * KL_freebits + ratio hinge (m vs diag(A))
         + latent guardrails (decorrelation + consistency).
- OOD eval: compare Jacobi vs VAE-Jacobi on cond numbers and 5-step PCG
  relative residual reduction.

Author: you :)
"""

import os, glob, math, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt
import pandas as pd

# =======================
# Shard helpers & band IO
# =======================

def list_shards(folder: str):
    return sorted(glob.glob(os.path.join(folder, "shard_*.pt")))

def load_shard(path: str):
    return torch.load(path, map_location="cpu")  # list of dicts

def payload_to_csr(d):
    crow = d["A_crow_indices"].cpu().numpy()
    col  = d["A_col_indices"].cpu().numpy()
    val  = d["A_values"].cpu().numpy()
    n0, n1 = d["A_shape"]
    return csr_matrix((val, col, crow), shape=(n0, n1))

def build_band_from_csr(A_csr: csr_matrix, S: int, W: int, dtype=np.float32):
    n = S*S
    X = np.zeros((2*W + 1, n), dtype=dtype)
    indptr, indices, data = A_csr.indptr, A_csr.indices, A_csr.data
    for r in range(n):
        start, end = indptr[r], indptr[r+1]
        for k in range(start, end):
            c = indices[k]; v = data[k]
            off = c - r
            if -W <= off <= W:
                X[off + W, r] = v
    return X  # [C, L]

def normalize_band(X, W):
    diag_idx = W
    diag_abs = np.abs(X[diag_idx])
    scale = np.median(diag_abs) + 1e-8
    Xn = (X / scale).astype(X.dtype, copy=False)
    return Xn, np.float32(scale)

def get_S_from_payload(d):
    n = int(d["A_shape"][0])
    S = int(round(math.sqrt(n)))
    assert S*S == n, f"Matrix is not S^2 x S^2 (n={n})"
    return S

# =======================
# Dataset
# =======================

class ShardedBandIterable(IterableDataset):
    """
    Streams (normalized) band inputs and the original SciPy CSR per sample.
    Yields dict with:
      X: torch.Tensor [C=2W+1, L=S^2] (normalized)
      scale: float (used to scale A consistently)
      Acsr: SciPy CSR (unscaled, CPU)
      S: int
      meta: dict
    """
    def __init__(self, folder: str, W: int, use_float32: bool = True,
                 limit_per_shard: int = None, shuffle_within_shard: bool = False, seed: int = 123):
        super().__init__()
        self.folder = folder
        self.W = W
        self.use_float32 = use_float32
        self.limit_per_shard = limit_per_shard
        self.shuffle_within_shard = shuffle_within_shard
        self.seed = seed

    def __iter__(self):
        shard_paths = list_shards(self.folder)
        for spath in shard_paths:
            payload = load_shard(spath)
            idxs = list(range(len(payload)))
            if self.shuffle_within_shard:
                g = torch.Generator()
                g.manual_seed(abs(hash((spath, self.seed))) % (2**31))
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]
            if self.limit_per_shard is not None:
                idxs = idxs[:self.limit_per_shard]

            for i in idxs:
                d = payload[i]
                S = get_S_from_payload(d)
                A = payload_to_csr(d)
                dtype = np.float32 if self.use_float32 else np.float64
                X = build_band_from_csr(A, S, self.W, dtype=dtype)
                X, scale = normalize_band(X, self.W)
                yield {
                    "X": torch.from_numpy(X),
                    "scale": float(scale),
                    "Acsr": A,
                    "S": S,
                    "meta": d["meta"]
                }

def collate_stream(batch):
    return batch[0] if len(batch) == 1 else {
        "X": [b["X"] for b in batch],
        "scale": [b["scale"] for b in batch],
        "Acsr": [b["Acsr"] for b in batch],
        "S": [b["S"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }

# =======================
# Model: VAE → diag(M)
# =======================

class BandVAE(nn.Module):
    """
    1D Conv VAE over [B, C=2W+1, L=S^2].
    Decoder predicts s = log(m/a_ii); final diag is m = a_ii * softplus(s) + eps.
    """
    def __init__(self, in_ch: int, latent: int = 64, base: int = 64, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, base,   7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(base, base*2,  7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(base*2, base*4,7, stride=2, padding=3), nn.GELU(),
        )
        self.mu = nn.Linear(base*4, latent)
        self.lv = nn.Linear(base*4, latent)

        self.dec_fc = nn.Linear(latent, base*4)
        self.dec_up = nn.Sequential(
            nn.ConvTranspose1d(base*4, base*2, 8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(base*2, base,   8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(base,   base,   8, stride=2, padding=3), nn.GELU(),
        )
        # Predict s = log(m/a_ii). Init near 0 => Jacobi baseline.
        self.to_s = nn.Conv1d(base, 1, kernel_size=1)
        nn.init.zeros_(self.to_s.weight); nn.init.zeros_(self.to_s.bias)

        # Tiny smoothing to prevent spiky s; starts as identity.
        self.smooth = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False, groups=1)
        with torch.no_grad():
            self.smooth.weight.zero_()
            self.smooth.weight[..., 2] = 1.0

    def encode(self, x):
        h = self.enc(x).mean(dim=2)
        mu = self.mu(h)
        lv = self.lv(h).clamp(-10.0, 10.0)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std

    def decode_s(self, z, L_out):
        h = self.dec_fc(z)[:, :, None]
        y = self.dec_up(h)
        if y.size(-1) < L_out:
            y = F.pad(y, (0, L_out - y.size(-1)))
        y = y[..., :L_out]
        s = self.to_s(y)                   # [B,1,L]
        s = self.smooth(s).squeeze(1)      # [B,L]
        return s

    def forward(self, x, a_diag_scaled=None, scale: float = 1.0):
        """
        x: [B,C,L] normalized band; a_diag_scaled: [B,L] (diag channel of x); scale rescales back.
        """
        B, C, L = x.shape
        assert a_diag_scaled is not None, "Pass the diag channel as a_diag_scaled"
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        s = self.decode_s(z, L_out=L)                # log-ratio
        a_diag = a_diag_scaled * scale               # unnormalize diag(A)
        m_diag = a_diag * F.softplus(s) + self.eps   # SPD by construction
        return m_diag, mu, lv

    @torch.no_grad()
    def infer_m_from_mu(self, x, a_diag_scaled, scale: float):
        mu, lv = self.encode(x)
        s = self.decode_s(mu, L_out=x.shape[-1])
        a_diag = a_diag_scaled * scale
        return a_diag * F.softplus(s) + self.eps

# =======================
# Loss pieces
# =======================

def kl_gauss_freebits(mu, lv, free_bits=0.02):
    """
    Computes the KL divergence between a diagonal Gaussian posterior N(mu, exp(lv)) and the standard normal prior,
    with optional free bits thresholding to prevent KL collapse.
    """
    kl_dim = 0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv)
    if free_bits > 0:
        kl_dim = torch.clamp(kl_dim, min=free_bits)
    return kl_dim.sum(dim=1).mean()

def _probe_bank(L, probes, device, dtype, seed=123):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    v = torch.randint(0, 2, (probes, L), generator=g, dtype=torch.int8).float()
    v[v==0] = -1.0
    return v.to(device=device, dtype=dtype)  # [P,L]

def precond_loss_diag_lowvar(M_diag, A_sparse, probes=4, seed=123):
    """
    E|| (I - M^{-1}A) v ||^2 using fixed Rademacher probes + antithetic pairing.
    """
    B, L = M_diag.shape
    V = _probe_bank(L, probes, M_diag.device, M_diag.dtype, seed=seed)
    loss = 0.0
    Minv = 1.0 / M_diag.clamp_min(1e-12)
    for p in range(probes):
        v = V[p].expand(B, -1)                  # [B,L]
        Av = torch.sparse.mm(A_sparse, v.T).T
        r1 = v - (Av * Minv)
        loss += r1.pow(2).sum(dim=1).mean()
    return loss / probes

def spectral_spread_loss(m_diag, A_sparse, probes=4, mean1_weight=1.0):
    """
    Computes the spectral spread loss for the preconditioned matrix B = M^{-1/2} A M^{-1/2}.
    The loss is:
        Var[rq] + mean1_weight * (E[rq] - 1)^2
    where rq is the Rayleigh quotient of random vectors.
    """
    Bsz, L = m_diag.shape
    inv_sqrt_m = (m_diag.clamp_min(1e-12)).rsqrt()
    R = []
    for _ in range(probes):
        v = torch.randn(Bsz, L, device=m_diag.device, dtype=m_diag.dtype)
        u = v * inv_sqrt_m
        Au = torch.sparse.mm(A_sparse, u.T).T
        Bu = Au * inv_sqrt_m
        rq = (u * Bu).sum(dim=1) / (u.pow(2).sum(dim=1) + 1e-12)
        R.append(rq)
    R = torch.stack(R, dim=1)      # [B, probes]
    rq_mean = R.mean(dim=1)
    rq_var  = R.var(dim=1, unbiased=False)
    return rq_var.mean() + mean1_weight * (rq_mean - 1.0).pow(2).mean()

def ratio_hinge_loss(m_diag: torch.Tensor, X_band: torch.Tensor, scale: float,
                     rmin: float = 1e-2, rmax: float = 1e+2):
    """
    Penalize log-ratio outside [log rmin, log rmax], where ratio = m_i / a_ii.
    Uses the diagonal channel from the normalized band to reconstruct a_ii.
    """
    # X_band: [B, C=2W+1, L], diag channel is at index W
    a_ii = X_band[:, X_band.size(1)//2, :] * scale  # [B, L]
    a_ii = a_ii.clamp_min(1e-30)                    # be safe
    log_ratio = (m_diag / a_ii).clamp_min(1e-30).log()  # log(m/a)
    lo, hi = math.log(rmin), math.log(rmax)
    loss_lo = (lo - log_ratio).clamp_min(0.0)
    loss_hi = (log_ratio - hi).clamp_min(0.0)
    return (loss_lo + loss_hi).mean()

def latent_cov_offdiag_penalty(mu: torch.Tensor, eps: float = 1e-5):
    B, D = mu.shape
    mu_c = mu - mu.mean(dim=0, keepdim=True)
    C = (mu_c.T @ mu_c) / (B - 1 + eps)
    off = C - torch.diag(torch.diag(C))
    return (off**2).mean()

def latent_consistency_penalty(encode_fn, X, noise_std=1e-3):
    noise = torch.randn_like(X) * noise_std
    mu1, _ = encode_fn(X)
    mu2, _ = encode_fn(X + noise)
    return F.mse_loss(mu1, mu2)

def beta_schedule(step, warmup_steps, cycle=0):
    if cycle <= 0:
        return 1.0 if warmup_steps <= 0 else min(1.0, (step+1)/warmup_steps)
    t = (step % cycle) / max(1, cycle-1)
    return 0.1 + 0.9 * t  # triangular 0.1 -> 1.0

def vae_precond_objective_with_ratio(
    m_diag, mu, lv, A_sparse, X_band, scale,
    probes=4,
    w_residual=1.0,
    w_spectral=1.0,          # stronger spectral shaping
    mean1_weight=1.0,
    beta=1.0,
    free_bits=0.02,
    w_ratio=1.0,
    rmin=1e-2,
    rmax=1e+2
):
    pre = precond_loss_diag_lowvar(m_diag, A_sparse, probes=probes)
    spec = spectral_spread_loss(m_diag, A_sparse, probes=probes, mean1_weight=mean1_weight)
    kl   = kl_gauss_freebits(mu, lv, free_bits=free_bits)
    rat  = ratio_hinge_loss(m_diag, X_band, scale, rmin=rmin, rmax=rmax)
    total = w_residual*pre + w_spectral*spec + beta*kl + w_ratio*rat
    return total, {
        "pre":  float(pre.detach().cpu()),
        "spec": float(spec.detach().cpu()),
        "kl":   float(kl.detach().cpu()),
        "ratio":float(rat.detach().cpu())
    }

# =======================
# Cond number + PCG utils
# =======================

def cond_estimate(A: csr_matrix, maxiter_eigs=100):
    try:
        lmax = eigsh(A, k=1, which='LM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        lmin = eigsh(A, k=1, which='SM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        if lmax <= 0 or lmin <= 0: return np.nan
        return float(lmax / lmin)
    except Exception:
        return np.nan

def cond_estimate_precond(A: csr_matrix, D: np.ndarray, maxiter_eigs=100):
    n = A.shape[0]
    Ds = np.sqrt(np.maximum(D, 1e-30))
    invDs = 1.0 / Ds
    def mv(x):
        y = x * invDs
        y = A @ y
        y = y * invDs
        return y
    Lop = LinearOperator((n, n), matvec=mv, dtype=A.dtype)
    try:
        lmax = eigsh(Lop, k=1, which='LM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        lmin = eigsh(Lop, k=1, which='SM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        if lmax <= 0 or lmin <= 0: return np.nan
        return float(lmax / lmin)
    except Exception:
        return np.nan

def pcg_5step_reduction(A: csr_matrix, b: np.ndarray, D: np.ndarray, steps: int = 5, tol_fallback=1e-32):
    """
    Run PCG for exactly `steps` iterations with diagonal preconditioner diag(D).
    Returns: rel_resid = ||r_k|| / ||r_0||  (k = steps), NaN-safe and warning-free.
    """
    n = A.shape[0]
    D = np.clip(D, 1e-20, 1e12)
    invD = 1.0 / D

    def matvec(x):
        return A @ x

    x = np.zeros(n, dtype=A.dtype)
    r = b - matvec(x)
    z = invD * r
    p = z.copy()

    r0_norm = np.linalg.norm(r)
    if not np.isfinite(r0_norm) or r0_norm < tol_fallback:
        return 1.0

    rz_old = float(np.dot(r, z))
    if not np.isfinite(rz_old) or abs(rz_old) < tol_fallback:
        return 1.0

    for _ in range(steps):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if not np.isfinite(pAp) or abs(pAp) < tol_fallback:
            break
        alpha = rz_old / pAp

        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = np.linalg.norm(r)
        if not np.isfinite(r_norm) or r_norm < tol_fallback:
            return 0.0

        z = invD * r
        rz_new = float(np.dot(r, z))
        if not np.isfinite(rz_new) or abs(rz_new) < tol_fallback:
            return r_norm / (r0_norm + 1e-30)

        beta = rz_new / (rz_old + 1e-30)
        p = z + beta * p
        rz_old = rz_new

    rel = np.linalg.norm(r) / (r0_norm + 1e-30)
    if not np.isfinite(rel):
        rel = 1.0
    return float(rel)

# =======================
# Training / OOD eval
# =======================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[Device] Using {device}")
    dtype = torch.float32 if args.use_float32 else torch.float64
    in_ch = 2*args.W + 1

    # Data
    train_ds = ShardedBandIterable(os.path.join(args.data_root, "train"),
                                   W=args.W, use_float32=args.use_float32,
                                   limit_per_shard=args.limit_per_shard,
                                   shuffle_within_shard=True, seed=123)
    ood_ds = ShardedBandIterable(os.path.join(args.data_root, "ood_test"),
                                 W=args.W, use_float32=args.use_float32,
                                 limit_per_shard=args.ood_limit_per_shard,
                                 shuffle_within_shard=False)
    train_loader = DataLoader(train_ds, batch_size=1, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers>0),
                              collate_fn=collate_stream)
    ood_loader = DataLoader(ood_ds, batch_size=1, num_workers=args.workers,
                            pin_memory=True, persistent_workers=(args.workers>0),
                            collate_fn=collate_stream)

    # Model + optim
    model = BandVAE(in_ch=in_ch, latent=args.latent, base=args.base).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)

    # ================= Train =================
    logs = []
    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            X = batch["X"].to(device=device, dtype=dtype, non_blocking=True)  # [C,L]
            if X.dim() == 2: X = X.unsqueeze(0)  # [1,C,L]
            scale = float(batch["scale"])

            # Build A in the SAME normalized domain (divide by scale)
            dA = batch["Acsr"]
            crow = torch.from_numpy(dA.indptr.astype(np.int64)).to(device)
            col  = torch.from_numpy(dA.indices.astype(np.int64)).to(device)
            val  = torch.from_numpy(dA.data / scale).to(device=device, dtype=dtype)
            L = dA.shape[0]
            A_t = torch.sparse_csr_tensor(crow, col, val, size=(L, L), dtype=dtype, device=device)
            A_sp = A_t  # CSR works well in torch

            # Forward (need diag channel)
            a_diag_scaled = X[:, X.size(1)//2, :]  # [B,L]
            m_diag, mu, lv = model(X, a_diag_scaled=a_diag_scaled, scale=scale)

            # Objective
            beta_now = beta_schedule(step, args.kl_warmup_steps, cycle=args.kl_cycle)
            loss, parts = vae_precond_objective_with_ratio(
                m_diag, mu, lv, A_sparse=A_sp, X_band=X, scale=scale,
                probes=args.probes,
                w_residual=args.w_residual,
                w_spectral=args.w_spectral,
                mean1_weight=args.mean1_weight,
                beta=beta_now,
                free_bits=args.free_bits,
                w_ratio=args.w_ratio,
                rmin=args.ratio_rmin,
                rmax=args.ratio_rmax
            )

            # Latent guardrails
            tc_pen  = latent_cov_offdiag_penalty(mu) * args.w_tc
            cons_pen = latent_consistency_penalty(lambda X_: model.encode(X_), X, noise_std=args.latent_noise_std) * args.w_consist

            loss = loss + tc_pen + cons_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            # Periodic debug of m range
            if (step % 100) == 0:
                mmin = float(m_diag.min().detach().cpu())
                mmax = float(m_diag.max().detach().cpu())
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 pre=f"{parts.get('pre', 0.0):.3f}",
                                 spec=f"{parts.get('spec', 0.0):.3f}",
                                 kl=f"{parts.get('kl', 0.0):.3f}",
                                 mmin=f"{mmin:.2e}", mmax=f"{mmax:.2e}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 pre=f"{parts.get('pre', 0.0):.3f}",
                                 spec=f"{parts.get('spec', 0.0):.3f}",
                                 kl=f"{parts.get('kl', 0.0):.3f}")

            logs.append({"step": step, "epoch": epoch, "loss": float(loss.item()),
                         "pre": parts["pre"], "spec": parts["spec"], "kl": parts["kl"]})
            step += 1

        # checkpoint
        ckpt_path = os.path.join(args.out_dir, f"vae_band_W{args.W}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

    # Plot training curve
    df = pd.DataFrame(logs)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(df["step"], df["loss"], label="total")
    if "pre" in df:  ax.plot(df["step"], df["pre"],  label="pre")
    if "spec" in df: ax.plot(df["step"], df["spec"], label="spec")
    if "kl" in df:   ax.plot(df["step"], df["kl"],   label="kl")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, f"train_curve_W{args.W}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"[Plot] Saved {fig_path}")

    # ================= OOD Eval =================
    model.eval()
    csv_path = os.path.join(args.out_dir, f"ood_eval_W{args.W}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "index","S","W",
            "cond_A","cond_Mjac_A","cond_Mvae_A",
            "rel_resid_5_jacobi","rel_resid_5_vae"
        ])

        idx = 0
        for batch in tqdm(ood_loader, desc="Evaluating OOD"):
            X = batch["X"].to(device=device, dtype=dtype, non_blocking=True)
            if X.dim() == 2: X = X.unsqueeze(0)
            S = int(batch["S"])
            Acsr = batch["Acsr"]
            scale = float(batch["scale"])

            with torch.no_grad():
                mu, lv = model.encode(X)
                a_diag_scaled_eval = X[:, X.size(1)//2, :]
                s_mu = model.decode_s(mu, L_out=X.shape[-1])           # [1,L]
                a_diag_np = (a_diag_scaled_eval * scale).squeeze(0).cpu().numpy()
                m_diag = (a_diag_np * F.softplus(s_mu).squeeze(0).cpu().numpy()) + model.eps

            # Safety clamps (eval-only) to avoid NaNs/Inf in fixed PCG
            D_vae = np.nan_to_num(m_diag, nan=1.0, posinf=1e12, neginf=1e-12)
            D_vae = np.clip(D_vae, 1e-20, 1e12)
            D_jac = np.clip(Acsr.diagonal().copy(), 1e-20, 1e12)

            # Condition numbers (robust to failure)
            condA    = cond_estimate(Acsr, maxiter_eigs=args.eigs_maxiter)
            condJacA = cond_estimate_precond(Acsr, D_jac, maxiter_eigs=args.eigs_maxiter)
            condVaeA = cond_estimate_precond(Acsr, D_vae, maxiter_eigs=args.eigs_maxiter)

            # Random unit rhs
            rng = np.random.default_rng(1234)
            b = rng.normal(0,1, size=Acsr.shape[0]); b /= (np.linalg.norm(b) + 1e-12)

            rel5_jac = pcg_5step_reduction(Acsr, b, D_jac, steps=5)
            rel5_vae = pcg_5step_reduction(Acsr, b, D_vae, steps=5)

            writer.writerow([
                idx, S, args.W,
                f"{condA:.6e}" if not np.isnan(condA) else "nan",
                f"{condJacA:.6e}" if not np.isnan(condJacA) else "nan",
                f"{condVaeA:.6e}" if not np.isnan(condVaeA) else "nan",
                f"{rel5_jac:.6e}", f"{rel5_vae:.6e}"
            ])
            idx += 1

    print(f"[CSV] OOD evaluation saved to {csv_path}")

    # Boxplot of 5-step residuals (log10 for readability)
    df_eval = pd.read_csv(csv_path)
    if len(df_eval):
        fig2, ax2 = plt.subplots(figsize=(6,4))
        jac = np.log10(np.maximum(df_eval["rel_resid_5_jacobi"].values, 1e-16))
        vae = np.log10(np.maximum(df_eval["rel_resid_5_vae"].values, 1e-16))
        ax2.boxplot([jac, vae], labels=["Jacobi", "VAE-Jacobi"])
        ax2.set_ylabel("log10 residual after 5 PCG steps")
        ax2.set_title(f"OOD: 5-step PCG residual reduction (W={args.W})")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2_path = os.path.join(args.out_dir, f"ood_pcg5_box_W{args.W}.png")
        fig2.savefig(fig2_path, dpi=150)
        print(f"[Plot] Saved {fig2_path}")

    print("[Done] Training + OOD evaluation complete.")

# =======================
# CLI
# =======================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_gen", help="Root with train/ and ood_test/ shards")
    ap.add_argument("--out_dir", type=str, default="runs_band", help="Output directory")
    ap.add_argument("--W", type=int, default=2, help="Band half-width (offsets -W..+W)")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--use_float32", action="store_true", help="Use float32 (default float64)")

    # Training objective knobs
    ap.add_argument("--probes", type=int, default=3, help="Hutchinson probe count (residual & spectral)")
    ap.add_argument("--w_residual", type=float, default=1.0, help="Weight for residual proxy")
    ap.add_argument("--w_spectral", type=float, default=1.0, help="Weight for spectral spread proxy")
    ap.add_argument("--mean1_weight", type=float, default=1.0, help="Weight for (E[rq]-1)^2")
    ap.add_argument("--free_bits", type=float, default=0.05, help="KL free-bits per latent dim (nats)")
    ap.add_argument("--kl_warmup_steps", type=int, default=1000, help="Linear warmup to beta=1")
    ap.add_argument("--kl_cycle", type=int, default=0, help="Cyclical β period (0 = disabled)")

    # Ratio hinge on m/diag(A)
    ap.add_argument("--w_ratio", type=float, default=1.0,
                    help="Weight for log(m/diag(A)) hinge outside [rmin,rmax]")
    ap.add_argument("--ratio_rmin", type=float, default=1e-2,
                    help="Lower bound on m/diag(A) (multiplicative)")
    ap.add_argument("--ratio_rmax", type=float, default=1e+2,
                    help="Upper bound on m/diag(A) (multiplicative)")

    # Latent guardrails
    ap.add_argument("--w_tc", type=float, default=5e-3, help="Weight for latent off-diagonal covariance penalty")
    ap.add_argument("--w_consist", type=float, default=5e-2, help="Weight for latent consistency penalty")
    ap.add_argument("--latent_noise_std", type=float, default=1e-3, help="Noise std for latent consistency")

    # Dataset limits
    ap.add_argument("--limit_per_shard", type=int, default=None, help="Train: limit samples per shard (debug)")
    ap.add_argument("--ood_limit_per_shard", type=int, default=None, help="Eval: limit samples per shard (debug)")

    # CPU linear algebra params
    ap.add_argument("--eigs_maxiter", type=int, default=300)

    # Legacy (unused by fixed-step evaluation but kept for completeness)
    ap.add_argument("--cg_tol", type=float, default=1e-6)
    ap.add_argument("--cg_maxiter", type=int, default=2000)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
