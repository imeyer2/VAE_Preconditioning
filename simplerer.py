#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae_jacobi_band_min.py — Minimal, NaN-safe VAE-Jacobi baseline

Goal: keep the exact same sharded dataset format, but use a very small VAE that
predicts a diagonal preconditioner M and show it's at least as good as (or
better than) plain Jacobi on OOD via a 5-step PCG reduction metric.

What this keeps:
- Same shard format (CSR pieces of A, plus meta), same band input of width W.
- Per-sample normalization by median |diag(A)|.
- Deterministic latent (no KL) for stability and simplicity.
- A single training loss (PCG-k) with small k (default 5).
- NaN/Inf guards and strict SPD check (drop bad samples) to avoid explosions.

What this removes (from your longer script):
- Eig min repair path, EMA tracking, multiple loss options, ratio guards, etc.
- Complex decoder transposed stacks; we upsample by linear interpolation.
- Extra logging and plots (we only write a tiny CSV + print summary).

Usage:
  python vae_jacobi_band_min.py \
    --data_root data_gen --out_dir runs_min --W 5 \
    --epochs 3 --lr 5e-4 --pcg_k 5 --workers 0

Outputs:
- runs_min/ood_eval_W{W}.csv  with per-sample 5-step relres for Identity, Jacobi, VAE
- A short printed summary of medians and win counts.
"""

import os, glob, math, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# -------------------------------
# Data helpers (band + normalize)
# -------------------------------

def list_shards(folder: str):
    return sorted(glob.glob(os.path.join(folder, "shard_*.pt")))

def load_shard(path: str):
    return torch.load(path, map_location="cpu")  # list[dict]

def payload_to_csr(d):
    crow = d["A_crow_indices"].cpu().numpy().astype(np.int64, copy=False)
    col  = d["A_col_indices"].cpu().numpy().astype(np.int64, copy=False)
    val  = d["A_values"].cpu().numpy().astype(np.float64, copy=False)
    n0, n1 = map(int, d["A_shape"])
    A = csr_matrix((val, col, crow), shape=(n0, n1), dtype=np.float64)
    return A

def get_S_from_payload(d):
    n = int(d["A_shape"][0])
    S = int(round(math.sqrt(n)))
    assert S*S == n, f"Matrix is not S^2 x S^2 (n={n})"
    return S

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
    return X

def normalize_band(X, W):
    diag = X[W]
    finite_diag = diag[np.isfinite(diag)]
    scale = np.median(np.abs(finite_diag)) if finite_diag.size else 1.0
    scale = float(scale) + 1e-8
    return (X / scale).astype(X.dtype, copy=False), scale

# ---------------
# SPD quick check
# ---------------

def is_spd(A: csr_matrix, sym_tol=1e-8, pd_tol=1e-12, maxiter=200):
    AT = A.transpose().tocsr()
    diff = (A - AT)
    sym_err = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    if sym_err > sym_tol:
        return False
    try:
        lmin = float(eigsh(A, k=1, which='SA', return_eigenvectors=False,
                           maxiter=maxiter, tol=1e-6)[0])
    except Exception:
        # Gershgorin fallback
        diag = A.diagonal()
        absA = A.copy(); absA.data = np.abs(absA.data)
        row_sums = np.array(absA.sum(axis=1)).ravel() - np.abs(diag)
        lmin = float(np.min(diag - row_sums))
    return (lmin > pd_tol)

# -------------
# Dataset
# -------------

class ShardedBand(IterableDataset):
    def __init__(self, folder: str, W: int, limit_per_shard: int | None = None,
                 shuffle=False, seed=123, sym_tol=1e-8, pd_tol=1e-12, eigs_maxiter=200):
        super().__init__()
        self.folder = folder
        self.W = W
        self.limit_per_shard = limit_per_shard
        self.shuffle = shuffle
        self.seed = seed
        self.sym_tol = sym_tol
        self.pd_tol = pd_tol
        self.eigs_maxiter = eigs_maxiter

    def __iter__(self):
        for spath in list_shards(self.folder):
            payload = load_shard(spath)
            idxs = list(range(len(payload)))
            if self.shuffle:
                g = torch.Generator(); g.manual_seed(abs(hash((spath, self.seed))) % (2**31))
                idxs = torch.randperm(len(idxs), generator=g).tolist()
            if self.limit_per_shard is not None:
                idxs = idxs[:self.limit_per_shard]
            for i in idxs:
                d = payload[i]
                S = get_S_from_payload(d)
                A = payload_to_csr(d)
                if not np.isfinite(A.data).all():
                    continue
                if not is_spd(A, sym_tol=self.sym_tol, pd_tol=self.pd_tol, maxiter=self.eigs_maxiter):
                    continue
                X = build_band_from_csr(A, S, self.W, dtype=np.float32)
                Xn, scale = normalize_band(X, self.W)
                if not np.isfinite(Xn).all() or not np.isfinite(scale):
                    continue
                yield {"X": torch.from_numpy(Xn), "scale": scale, "Acsr": A, "S": S, "meta": d.get("meta", {})}

# ---------
# Tiny VAE
# ---------

class TinyVAE(nn.Module):
    """A very small Conv1d VAE-like head with deterministic latent by default."""
    def __init__(self, in_ch: int, latent: int = 32, base: int = 32, eps: float = 1e-6, deterministic=True):
        super().__init__()
        self.eps = eps
        self.det = deterministic
        self.enc1 = nn.Conv1d(in_ch, base, 7, padding=3)
        self.enc2 = nn.Conv1d(base, base, 7, stride=2, padding=3)
        self.enc3 = nn.Conv1d(base, base, 7, stride=2, padding=3)
        self.mu = nn.Linear(base, latent)
        self.lv = nn.Linear(base, latent)
        self.fc = nn.Linear(latent, base)
        self.dec1 = nn.Conv1d(base, base, 3, padding=1)
        self.out = nn.Conv1d(base, 1, 1)
        nn.init.constant_(self.out.bias, math.log(math.e - 1.0))  # softplus ~ 1 at start (Jacobi)

    def encode(self, x):
        h = F.gelu(self.enc1(x))
        h = F.gelu(self.enc2(h))
        h = F.gelu(self.enc3(h))
        h = h.mean(dim=2)  # [B, C]
        mu = self.mu(h)
        lv = self.lv(h).clamp(-10, 10)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std

    def decode(self, z, L):
        h = F.gelu(self.fc(z))[:, :, None]  # [B, C, 1]
        y = F.gelu(self.dec1(h))
        # simple upsample to length L then refine
        y = F.interpolate(y, size=L, mode='linear', align_corners=False)
        s = self.out(y).squeeze(1)  # [B, L]
        return s

    def forward(self, x, a_diag_scaled):
        B, C, L = x.shape
        mu, lv = self.encode(x)
        z = mu if self.det else self.reparam(mu, lv)
        s = self.decode(z, L)
        m_diag_norm = a_diag_scaled * F.softplus(s) + self.eps
        return m_diag_norm

# -------------------------
# Hutchinson residual-proxy loss & eval
# -------------------------

def _probe_bank(L, probes, device, dtype, seed=123):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    v = torch.randint(0, 2, (probes, L), generator=g, dtype=torch.int8).float()
    v[v==0] = -1.0
    # Normalize by sqrt(L) so the estimator reflects a per-DOF scale
    v = v / math.sqrt(max(L, 1))
    return v.to(device=device, dtype=dtype)

def hutch_proxy_loss(A_t, m_diag_norm, probes=4, seed=123):
    """Per-DOF Hutchinson proxy for ||I - M^{-1}A||_F^2 / L.
    Lower is better. Uses the normalized domain (A/scale, M/scale).
    """
    B, L = m_diag_norm.shape
    V = _probe_bank(L, probes, m_diag_norm.device, m_diag_norm.dtype, seed=seed)
    Minv = 1.0 / m_diag_norm.clamp_min(1e-12)
    losses = []
    for p in range(probes):
        v = V[p].expand(B, -1)
        Av = torch.sparse.mm(A_t, v.T).T
        r = v - (Av * Minv)
        # mean over DOFs gives per-dof scale (stable across varying L)
        losses.append((r.pow(2).mean(dim=1)).mean())
    return torch.stack(losses).mean()

def hutch_proxy_for_fixed_diag(A_t, diag_norm, probes=4, seed=123):
    B, L = diag_norm.shape
    V = _probe_bank(L, probes, diag_norm.device, diag_norm.dtype, seed=seed)
    Minv = 1.0 / diag_norm.clamp_min(1e-12)
    losses = []
    for p in range(probes):
        v = V[p].expand(B, -1)
        Av = torch.sparse.mm(A_t, v.T).T
        r = v - (Av * Minv)
        losses.append((r.pow(2).mean(dim=1)).mean())
    return torch.stack(losses).mean()

# --------------
# Train & eval
# --------------


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    in_ch = 2*args.W + 1

    # Data
    train_ds = ShardedBand(os.path.join(args.data_root, "train"), W=args.W,
                           limit_per_shard=args.limit_per_shard, shuffle=True,
                           sym_tol=args.sym_tol, pd_tol=args.pd_tol, eigs_maxiter=args.eigs_maxiter)
    ood_ds   = ShardedBand(os.path.join(args.data_root, "ood_test"), W=args.W,
                           limit_per_shard=args.ood_limit_per_shard, shuffle=False,
                           sym_tol=args.sym_tol, pd_tol=args.pd_tol, eigs_maxiter=args.eigs_maxiter)

    def custom_collate(batch):
        # For batch_size=1, just return the single dict
        if len(batch) == 1:
            return batch[0]
        # For batch_size > 1, keep sparse matrices in a list
        out = {}
        keys = batch[0].keys()
        for k in keys:
            if isinstance(batch[0][k], torch.Tensor) and batch[0][k].ndim > 0:
                out[k] = torch.stack([b[k] for b in batch])
            elif isinstance(batch[0][k], (int, float, str)):
                out[k] = [b[k] for b in batch]
            else:
                out[k] = [b[k] for b in batch]
        return out

    train_loader = DataLoader(train_ds, batch_size=1, num_workers=args.workers,
                             pin_memory=True, persistent_workers=(args.workers>0),
                             collate_fn=custom_collate)
    ood_loader   = DataLoader(ood_ds, batch_size=1, num_workers=args.workers,
                             pin_memory=True, persistent_workers=(args.workers>0),
                             collate_fn=custom_collate)

    # Model + opt
    model = TinyVAE(in_ch=in_ch, latent=args.latent, base=args.base, deterministic=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Train ---
    # Debug flag: evaluate baselines on the first few batches of epoch 1
    debug_print_first_n = 3

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            X = batch["X"].to(device=device, dtype=dtype)
            if X.dim() == 2: X = X.unsqueeze(0)
            a_diag_scaled = X[:, args.W, :]
            scale = float(batch["scale"])  # not used in loss (A is scaled too)

            A = batch["Acsr"]
            crow = torch.from_numpy(A.indptr.astype(np.int64)).to(device)
            col  = torch.from_numpy(A.indices.astype(np.int64)).to(device)
            val  = torch.from_numpy((A.data/scale).astype(np.float32)).to(device)
            A_t = torch.sparse_csr_tensor(crow, col, val, size=A.shape, dtype=dtype, device=device)

            # --- DEBUG: compare baselines on the first few batches of epoch 1 ---
            if epoch == 1 and batch_idx < debug_print_first_n:
                with torch.no_grad():
                    L = a_diag_scaled.shape[1]
                    id_diag = torch.ones_like(a_diag_scaled)
                    jac_diag = a_diag_scaled.clamp_min(1e-20)
                    loss_id  = float(hutch_proxy_for_fixed_diag(A_t, id_diag,  probes=args.probes))
                    loss_jac = float(hutch_proxy_for_fixed_diag(A_t, jac_diag, probes=args.probes))
                print(f"[DEBUG] epoch{epoch} batch{batch_idx}: L={a_diag_scaled.shape[1]}, scale={scale:.3e}, baselines -> ID={loss_id:.3e}, Jac={loss_jac:.3e}")

            # forward & loss
            m_diag_norm = model(X, a_diag_scaled).clamp_min(1e-20)
            loss = hutch_proxy_loss(A_t, m_diag_norm, probes=args.probes, seed=123)

            # --- DEBUG: how far from Jacobi? ratio stats
            if epoch == 1 and batch_idx < debug_print_first_n:
                with torch.no_grad():
                    ratio = (m_diag_norm / a_diag_scaled.clamp_min(1e-20)).detach().cpu()
                    r_mean = ratio.mean().item(); r_std = ratio.std().item()
                    print(f"[DEBUG] epoch{epoch} batch{batch_idx}: m/diag mean={r_mean:.3f}, std={r_std:.3f}, loss={float(loss.detach().cpu()):.3e}")

            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # --- DEBUG: gradient norm ---
            if epoch == 1 and batch_idx < debug_print_first_n:
                total_gn = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_gn += float(p.grad.detach().norm().cpu())
                print(f"[DEBUG] epoch{epoch} batch{batch_idx}: grad_norm_sum={total_gn:.3e}")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # save checkpoint each epoch (tiny)
        torch.save({"model": model.state_dict(), "args": vars(args)},
                   os.path.join(args.out_dir, f"tinyvae_W{args.W}_e{epoch}.pt"))

    # --- OOD eval ---
    model.eval()
    csv_path = os.path.join(args.out_dir, f"ood_eval_W{args.W}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["idx","S","hutch_id","hutch_jac","hutch_vae"])
        idx = 0
        for batch in tqdm(ood_loader, desc="OOD eval"):
            X = batch["X"].to(device=device, dtype=dtype)
            if X.dim() == 2: X = X.unsqueeze(0)
            a_diag_scaled = X[:, args.W, :]
            scale = float(batch["scale"])  
            A = batch["Acsr"]

            with torch.no_grad():
                m_diag_norm = model(X, a_diag_scaled).clamp_min(1e-20)
                m = (m_diag_norm.squeeze(0).detach().cpu().numpy()) * scale
            if not np.isfinite(m).all():
                m = A.diagonal().copy()
            # Build normalized torch CSR again for eval of Hutchinson proxy
            crow = torch.from_numpy(A.indptr.astype(np.int64)).to(device)
            col  = torch.from_numpy(A.indices.astype(np.int64)).to(device)
            val  = torch.from_numpy((A.data/scale).astype(np.float32)).to(device)
            A_t = torch.sparse_csr_tensor(crow, col, val, size=A.shape, dtype=dtype, device=device)

            with torch.no_grad():
                # VAE diag (normalized domain)
                m_diag_norm = model(X, a_diag_scaled).clamp_min(1e-20)
                # Identity and Jacobi diagonals in normalized domain
                id_diag  = torch.ones_like(a_diag_scaled)
                jac_diag = a_diag_scaled.clamp_min(1e-20)

                h_id  = float(hutch_proxy_for_fixed_diag(A_t, id_diag,  probes=args.probes))
                h_jac = float(hutch_proxy_for_fixed_diag(A_t, jac_diag, probes=args.probes))
                h_vae = float(hutch_proxy_loss(A_t, m_diag_norm, probes=args.probes))

            wr.writerow([idx, int(batch["S"]), f"{h_id:.6e}", f"{h_jac:.6e}", f"{h_vae:.6e}"])
            idx += 1

    # quick summary
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        df = pd.read_csv(csv_path)
        hid  = df["hutch_id"].astype(float).values
        hj   = df["hutch_jac"].astype(float).values
        hv   = df["hutch_vae"].astype(float).values
        print("=== OOD Hutchinson proxy (lower is better) ===")
        print(f"median Identity: {np.median(hid):.3e}")
        print(f"median Jacobi  : {np.median(hj):.3e}")
        print(f"median VAE     : {np.median(hv):.3e}")
        wins = (hv <= hj).sum()
        print(f"VAE beats-or-ties Jacobi on {wins}/{len(hv)} samples")
        print(f"CSV: {csv_path}")

        # --- Visualization 1: Box and whisker plot ---
        fig, ax = plt.subplots(figsize=(6,4))
        ax.boxplot([hid, hj, hv], labels=["Identity","Jacobi","VAE"], showmeans=True)
        ax.set_ylabel("Hutchinson proxy (lower=better)")
        ax.set_title("Distribution across samples")
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"boxplot_hutch_W{args.W}.png"), dpi=150)

        # --- Visualization 2: Improvement scatter (Jacobi vs VAE) ---
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(hj, hv, alpha=0.6)
        ax.plot([min(hj.min(), hv.min()), max(hj.max(), hv.max())], [min(hj.min(), hv.min()), max(hj.max(), hv.max())], 'r--')
        ax.set_xlabel("Jacobi proxy")
        ax.set_ylabel("VAE proxy")
        ax.set_title("Sample-wise Hutchinson comparison")
        ax.set_aspect('equal','box')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"scatter_jac_vs_vae_W{args.W}.png"), dpi=150)

        # --- Visualization 3: Hint-style histogram of improvements ---
        improvement = (hj - hv)/hj
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(improvement*100, bins=30, alpha=0.7, color='tab:blue')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel("Percent improvement vs Jacobi (%)")
        ax.set_ylabel("# samples")
        ax.set_title("Histogram of VAE gain over Jacobi")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"hist_improvement_W{args.W}.png"), dpi=150)
        print("[Plot] Saved boxplot, scatter, and histogram to output directory.")
    except Exception as e:
        print(f"[WARN] Could not summarize CSV: {e}")

# -----------
# CLI
# -----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_gen")
    ap.add_argument("--out_dir", type=str, default="runs_min")
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--probes", type=int, default=8)  # a bit higher by default for stability
    ap.add_argument("--limit_per_shard", type=int, default=None)
    ap.add_argument("--ood_limit_per_shard", type=int, default=None)
    ap.add_argument("--sym_tol", type=float, default=1e-8)
    ap.add_argument("--pd_tol", type=float, default=1e-12)
    ap.add_argument("--eigs_maxiter", type=int, default=200)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
