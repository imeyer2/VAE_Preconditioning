#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae_band_minimal.py

A **minimal** VAE that learns a diagonal preconditioner M for SPD matrices (S^2 x S^2)
from a contiguous band of width W (offsets -W..+W).

- Input per sample: X in R^{(2W+1) x S^2}, normalized by per-sample scale (median |diag|).
- Decoder outputs m_diag (length S^2), with M = diag(softplus(...)+eps) > 0.
- Loss = Hutchinson residual proxy E|| (I - M^{-1}A) v ||^2 + beta * KL(N(mu,exp(lv)) || N(0,I)).
- OOD eval: compare Jacobi vs VAE-Jacobi on condition numbers and PCG iterations.

Designed to be as short and clear as possible.
"""

import os, glob, math, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh, cg
from tqdm import tqdm

# -----------------------------
# Shard helpers & band features
# -----------------------------

def list_shards(folder: str):
    return sorted(glob.glob(os.path.join(folder, "shard_*.pt")))

def load_shard(path: str):
    return torch.load(path, map_location="cpu")  # list[dict]

def payload_to_csr(d):
    crow = d["A_crow_indices"].cpu().numpy()
    col  = d["A_col_indices"].cpu().numpy()
    val  = d["A_values"].cpu().numpy()
    n0, n1 = d["A_shape"]
    return csr_matrix((val, col, crow), shape=(n0, n1))

def get_S_from_payload(d):
    n = int(d["A_shape"][0])
    S = int(round(math.sqrt(n)))
    assert S*S == n, f"Matrix is not S^2 x S^2 (n={n})"
    return S

def build_band_from_csr(A: csr_matrix, S: int, W: int, dtype=np.float32):
    n = S*S
    X = np.zeros((2*W + 1, n), dtype=dtype)
    indptr, indices, data = A.indptr, A.indices, A.data
    for r in range(n):
        start, end = indptr[r], indptr[r+1]
        for k in range(start, end):
            c = indices[k]; v = data[k]
            off = c - r
            if -W <= off <= W:
                X[off + W, r] = v
    return X

def normalize_band(X, W):
    diag_abs = np.abs(X[W])
    scale = np.median(diag_abs) + 1e-8
    return (X / scale).astype(X.dtype, copy=False), float(scale)


def identity_collate(batch):
    return batch[0]  # for batch_size=1



# ---------
# Dataset
# ---------

class ShardedBandStream(IterableDataset):
    """Streams normalized band inputs + SciPy CSR per sample."""
    def __init__(self, folder: str, W: int, use_float32: bool = True, limit_per_shard: int | None = None):
        super().__init__()
        self.folder, self.W, self.use_float32, self.limit_per_shard = folder, W, use_float32, limit_per_shard

    def __iter__(self):
        for spath in list_shards(self.folder):
            payload = load_shard(spath)
            idxs = range(len(payload)) if self.limit_per_shard is None else range(min(self.limit_per_shard, len(payload)))
            for i in idxs:
                d = payload[i]
                S = get_S_from_payload(d)
                A = payload_to_csr(d)
                dtype = np.float32 if self.use_float32 else np.float64
                X = build_band_from_csr(A, S, self.W, dtype)
                X, scale = normalize_band(X, self.W)
                yield {"X": torch.from_numpy(X), "scale": scale, "Acsr": A, "S": S}

# ---------
# Model
# ---------

class BandVAE(nn.Module):
    """Tiny 1D Conv VAE over [B, C=2W+1, L=S^2]; decoder → diag(M)."""
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
        self.to_diag = nn.Conv1d(base, 1, kernel_size=1)
        nn.init.zeros_(self.to_diag.weight); nn.init.zeros_(self.to_diag.bias)

    def encode(self, x):  # x:[B,C,L]
        h = self.enc(x).mean(dim=2)
        mu = self.mu(h)
        lv = self.lv(h).clamp(-10.0, 10.0)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std

    def decode_diag(self, z, L_out):
        h = self.dec_fc(z)[:, :, None]
        y = self.dec_up(h)
        if y.size(-1) < L_out:
            y = F.pad(y, (0, L_out - y.size(-1)))
        y = y[..., :L_out]
        m_raw = self.to_diag(y).squeeze(1)
        return F.softplus(m_raw) + self.eps

    def forward(self, x):
        B, C, L = x.shape
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        m = self.decode_diag(z, L)
        return m, mu, lv

# -----------------
# Loss components
# -----------------

def kl_gauss(mu, lv):
    return 0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv).sum(dim=1).mean()

def hutchinson_residual(M_diag, A_sparse, probes=2):
    """E || (I - M^{-1}A) v ||^2 with v ~ N(0,I)."""
    B, L = M_diag.shape
    loss = 0.0
    for _ in range(probes):
        v = torch.randn(B, L, device=M_diag.device, dtype=M_diag.dtype)
        Av = torch.sparse.mm(A_sparse, v.T).T
        MinvAv = Av / M_diag.clamp_min(1e-12)
        r = v - MinvAv
        loss = loss + (r.pow(2).sum(dim=1).mean())
    return loss / probes

# -------------------------
# Cond number + PCG utils
# -------------------------

def cond_estimate(A: csr_matrix, maxiter_eigs=200):
    try:
        lmax = eigsh(A, k=1, which='LM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        lmin = eigsh(A, k=1, which='SM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        if lmax <= 0 or lmin <= 0: return np.nan
        return float(lmax / lmin)
    except Exception:
        return np.nan

def cond_estimate_precond(A: csr_matrix, D: np.ndarray, maxiter_eigs=200):
    n = A.shape[0]
    Ds = np.sqrt(np.maximum(D, 1e-30)); invDs = 1.0 / Ds
    def mv(x):
        y = x * invDs; y = A @ y; y = y * invDs
        return y
    Lop = LinearOperator((n, n), matvec=mv, dtype=A.dtype)
    try:
        lmax = eigsh(Lop, k=1, which='LM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        lmin = eigsh(Lop, k=1, which='SM', return_eigenvectors=False, maxiter=maxiter_eigs)[0]
        if lmax <= 0 or lmin <= 0: return np.nan
        return float(lmax / lmin)
    except Exception:
        return np.nan

def pcg_iters(A: csr_matrix, b: np.ndarray, D: np.ndarray, tol=1e-6, maxiter=2000):
    invD = 1.0 / np.maximum(D, 1e-30)
    M = LinearOperator(A.shape, matvec=lambda r: invD * r, dtype=A.dtype)
    iters = {"k": 0}
    def cb(_): iters["k"] += 1
    _, info = cg(A, b, M=M, rtol=tol, maxiter=maxiter, callback=cb)
    return iters["k"] if info == 0 else max(iters["k"], info if isinstance(info, int) else maxiter)

# -------------
# Training
# -------------

def train_and_eval(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch.float32 if args.use_float32 else torch.float64
    in_ch = 2*args.W + 1

    # Data
    train_ds = ShardedBandStream(os.path.join(args.data_root, "train"), W=args.W, use_float32=args.use_float32, limit_per_shard=args.limit_per_shard)
    ood_ds   = ShardedBandStream(os.path.join(args.data_root, "ood_test"), W=args.W, use_float32=args.use_float32, limit_per_shard=args.ood_limit_per_shard)
    train_loader = DataLoader(train_ds, batch_size=1, num_workers=0, collate_fn=identity_collate)
    ood_loader   = DataLoader(ood_ds,   batch_size=1, num_workers=0, collate_fn=identity_collate)


    # Model/optim
    model = BandVAE(in_ch=in_ch, latent=args.latent, base=args.base).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    # Train
    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            X = batch["X"].to(device=device, dtype=dtype).unsqueeze(0)  # [1,C,L]
            scale = float(batch["scale"])  # used only to keep A in same normalized domain

            A = batch["Acsr"]
            crow = torch.from_numpy(A.indptr.astype(np.int64)).to(device)
            col  = torch.from_numpy(A.indices.astype(np.int64)).to(device)
            val  = torch.from_numpy((A.data / scale)).to(device=device, dtype=dtype)
            L = A.shape[0]
            A_sp = torch.sparse_csr_tensor(crow, col, val, size=(L, L), dtype=dtype, device=device)

            m, mu, lv = model(X)
            pre = hutchinson_residual(m, A_sp, probes=args.probes)
            kl  = kl_gauss(mu, lv)
            loss = pre + args.beta * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}", pre=f"{pre.item():.4f}", kl=f"{kl.item():.4f}")
            step += 1

        # Save tiny checkpoint
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, f"vae_band_min_W{args.W}_epoch{epoch}.pt"))

    # -----------------
    # OOD evaluation
    # -----------------
    model.eval()
    csv_path = os.path.join(args.out_dir, f"ood_eval_W{args.W}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index","S","W","cond_A","cond_Mjac_A","cond_Mvae_A","cg_iters_jacobi","cg_iters_vae_jacobi"]) 
        idx = 0
        for batch in tqdm(ood_loader, desc="Evaluating OOD"):
            X = batch["X"].to(device=device, dtype=dtype).unsqueeze(0)
            S = int(batch["S"]) ; A = batch["Acsr"] ; scale = float(batch["scale"]) 

            with torch.no_grad():
                mu, lv = model.encode(X)
                m_norm = model.decode_diag(mu, L_out=X.shape[-1])  # deterministic (use mu)
            m_diag = (m_norm.squeeze(0).cpu().numpy()) * scale
            D_vae = np.maximum(m_diag, 1e-12)
            D_jac = np.maximum(A.diagonal().copy(), 1e-12)

            condA    = cond_estimate(A, maxiter_eigs=args.eigs_maxiter)
            condJacA = cond_estimate_precond(A, D_jac, maxiter_eigs=args.eigs_maxiter)
            condVaeA = cond_estimate_precond(A, D_vae, maxiter_eigs=args.eigs_maxiter)

            rng = np.random.default_rng(1234)
            b = rng.normal(0,1, size=A.shape[0]); b /= (np.linalg.norm(b) + 1e-12)
            it_jac = pcg_iters(A, b, D_jac, tol=args.cg_tol, maxiter=args.cg_maxiter)
            it_vja = pcg_iters(A, b, D_vae, tol=args.cg_tol, maxiter=args.cg_maxiter)

            writer.writerow([idx, S, args.W,
                             f"{condA:.6e}" if not np.isnan(condA) else "nan",
                             f"{condJacA:.6e}" if not np.isnan(condJacA) else "nan",
                             f"{condVaeA:.6e}" if not np.isnan(condVaeA) else "nan",
                             it_jac, it_vja])
            idx += 1

    print(f"[CSV] OOD evaluation saved to {csv_path}")

# -------------
# CLI
# -------------

def parse_args():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--data_root", type=str, default="data_gen")
    ap.add_argument("--out_dir",   type=str, default="runs_min")
    # core setup
    ap.add_argument("--W",       type=int,   default=2)
    ap.add_argument("--latent",  type=int,   default=64)
    ap.add_argument("--base",    type=int,   default=64)
    ap.add_argument("--epochs",  type=int,   default=5)
    ap.add_argument("--lr",      type=float, default=5e-4)
    ap.add_argument("--beta",    type=float, default=1.0, help="KL weight")
    ap.add_argument("--device",  type=str,   default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--use_float32", action="store_true")
    # sampling + eval
    ap.add_argument("--probes",        type=int,   default=2)
    ap.add_argument("--eigs_maxiter",  type=int,   default=200)
    ap.add_argument("--cg_tol",        type=float, default=1e-6)
    ap.add_argument("--cg_maxiter",    type=int,   default=2000)
    # debug
    ap.add_argument("--limit_per_shard",     type=int, default=None)
    ap.add_argument("--ood_limit_per_shard", type=int, default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_and_eval(args)
