#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae_tridiag_ldlt.py

VAE that learns an SPD **tridiagonal** preconditioner via an LDLᵀ factorization.
Training is aligned in the normalized domain (A/scale). Evaluation runs PCG-5
with Identity vs Jacobi vs VAE-Tridiag.

Key ideas:
- Decoder outputs (s_d, t). We set:
    d_i  = diag(A)/scale * softplus(s_d)_i + eps          (length L, positive)
    ell_i = tanh(t_i)                                     (length L-1, unitless)
  Then M = L D Lᵀ (L unit lower bidiagonal with ell on subdiagonal) is SPD.
- Apply M^{-1} with two triangular solves + diagonal scaling (O(L)), fully
  differentiable, stable, and batch-friendly.

Usage:
  python vae_tridiag_ldlt.py --data_root data_gen --out_dir runs_tridiag --W 5
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
import pandas as pd
import matplotlib.pyplot as plt

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
    return csr_matrix((val, col, crow), shape=(n0, n1), dtype=np.float64)

def get_S_from_payload(d):
    n = int(d["A_shape"][0])
    S = int(round(math.sqrt(n)))
    assert S*S == n, f"Matrix is not S^2 x S^2 (n={n})"
    return S

def build_band_from_csr(A_csr: csr_matrix, S: int, W: int, dtype=np.float32):
    """Return X ∈ R^{(2W+1) x S^2} with offsets [-W..W]."""
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
    """Normalize by median |diag| so scales are comparable."""
    diag = X[W]
    finite_diag = diag[np.isfinite(diag)]
    scale = np.median(np.abs(finite_diag)) if finite_diag.size else 1.0
    scale = float(scale) + 1e-8
    return (X/scale).astype(X.dtype, copy=False), scale

# -----------------
# SPD sanity checks
# -----------------

def _gershgorin_bounds(A: csr_matrix):
    diag = A.diagonal()
    absA = A.copy(); absA.data = np.abs(absA.data)
    row_sums = np.array(absA.sum(axis=1)).ravel() - np.abs(diag)
    gmin = float(np.min(diag - row_sums))
    gmax = float(np.max(diag + row_sums))
    return gmin, gmax

def _eig_min(A: csr_matrix, maxiter=200, tol=1e-6):
    try:
        return float(eigsh(A, k=1, which='SA', return_eigenvectors=False,
                           maxiter=maxiter, tol=tol)[0])
    except Exception:
        gmin, _ = _gershgorin_bounds(A)
        return gmin

def is_spd(A: csr_matrix, sym_tol=1e-8, pd_tol=1e-12, maxiter=200):
    AT = A.transpose().tocsr()
    diff = (A - AT)
    sym_err = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    if sym_err > sym_tol:
        return False, {"sym_err": sym_err, "lmin": np.nan}
    lmin = _eig_min(A, maxiter=maxiter)
    return (lmin > pd_tol), {"sym_err": sym_err, "lmin": lmin}

# -------------
# Dataset
# -------------

class ShardedBandIterable(IterableDataset):
    """
    Streams (X, scale, A, S) from shards with SPD checks.
    Yields:
      X: torch.FloatTensor [C=2W+1, L=S^2] (normalized)
      scale: float
      Acsr: SciPy CSR (unscaled, float64)
      S: int
      meta: dict
    """
    def __init__(self, folder: str, W: int, limit_per_shard: int = None,
                 shuffle=False, seed=123, strict_spd_check=True,
                 sym_tol=1e-8, pd_tol=1e-12, eigs_maxiter=200):
        super().__init__()
        self.folder = folder; self.W = W
        self.limit_per_shard = limit_per_shard
        self.shuffle = shuffle; self.seed = seed
        self.strict = strict_spd_check
        self.sym_tol = sym_tol; self.pd_tol = pd_tol; self.eigs_maxiter = eigs_maxiter

    def __iter__(self):
        shard_paths = list_shards(self.folder)
        for spath in shard_paths:
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
                ok, info = is_spd(A, sym_tol=self.sym_tol, pd_tol=self.pd_tol, maxiter=self.eigs_maxiter)
                if not ok:
                    print(f"[SPD-FAIL] shard={os.path.basename(spath)} idx={i} sym_err={info['sym_err']:.3e} lmin={info['lmin']:.3e} -> dropped")
                    continue
                X = build_band_from_csr(A, S, self.W, dtype=np.float32)
                Xn, scale = normalize_band(X, self.W)
                if not np.isfinite(Xn).all() or not np.isfinite(scale):
                    continue
                yield {
                    "X": torch.from_numpy(Xn),  # [C, L] normalized
                    "scale": scale,
                    "Acsr": A,                  # unnormalized SciPy CSR
                    "S": S,
                    "meta": d.get("meta", {})
                }

# -------------
# VAE (tridiagonal LDLᵀ)
# -------------

def _inv_softplus_one():
    # softplus(b) = 1  => b = log(exp(1)-1)
    return math.log(math.e - 1.0)

class BandVAE_LDLT(nn.Module):
    """
    Encoder-decoder VAE that outputs tridiagonal SPD preconditioner M via LDLᵀ.

    Decoder heads:
      - to_sd -> s_d: length L  (scales diag(A)/scale via softplus to get d>0)
      - to_t  -> t:   length L  (we'll use t[:-1] and map with tanh to ell in (-1,1))

    forward(): returns (d_norm, ell) for A/scale domain (and mu, lv)
    infer_from_mu(): returns (d_unnorm, ell) for unnormalized domain (d*scale)
    """
    def __init__(self, in_ch: int, latent: int = 64, base: int = 64, eps: float = 1e-6,
                 deterministic_latent: bool = True, ell_clip: float = 0.98):
        super().__init__()
        self.eps = eps
        self.deterministic = deterministic_latent
        self.ell_clip = float(ell_clip)

        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, base,    7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(base, base*2,   7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(base*2, base*4, 7, stride=2, padding=3), nn.GELU(),
        )
        self.mu = nn.Linear(base*4, latent)
        self.lv = nn.Linear(base*4, latent)

        self.dec_fc = nn.Linear(latent, base*4)
        self.dec_up = nn.Sequential(
            nn.ConvTranspose1d(base*4, base*2, 8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(base*2, base,   8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(base,   base,   8, stride=2, padding=3), nn.GELU(),
        )
        self.to_sd = nn.Conv1d(base, 1, kernel_size=1)  # s_d
        self.to_t  = nn.Conv1d(base, 1, kernel_size=1)  # t

        nn.init.zeros_(self.to_sd.weight); nn.init.zeros_(self.to_sd.bias)
        nn.init.zeros_(self.to_t.weight);  nn.init.zeros_(self.to_t.bias)
        # Start exactly at Jacobi: softplus(s_d)=1 => d_norm = diag(A)/scale
        with torch.no_grad():
            self.to_sd.bias.fill_(_inv_softplus_one())
            self.to_t.bias.zero_()  # ell = tanh(0) = 0 -> diagonal preconditioner initially

    def encode(self, x):
        h = self.enc(x).mean(dim=2)
        mu = self.mu(h)
        lv = self.lv(h).clamp(-10.0, 10.0)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std

    def _decode_heads(self, z, L_out):
        h = self.dec_fc(z)[:, :, None]
        y = self.dec_up(h)
        if y.size(-1) < L_out:
            y = F.pad(y, (0, L_out - y.size(-1)))
        y = y[..., :L_out]
        s_d = self.to_sd(y).squeeze(1)          # [B, L]
        t   = self.to_t(y).squeeze(1)           # [B, L]
        return s_d, t

    def forward(self, x, a_diag_scaled, scale: float):
        """
        Return (d_norm, ell) for A/scale domain.
        d_norm: [B,L] positive
        ell:    [B,L-1] in (-ell_clip, ell_clip) via tanh
        """
        B, C, L = x.shape
        mu, lv  = self.encode(x)
        z       = mu if self.deterministic else self.reparam(mu, lv)
        s_d, t  = self._decode_heads(z, L_out=L)

        d_norm  = a_diag_scaled * F.softplus(s_d) + self.eps
        ell     = torch.tanh(t)[..., :-1] * self.ell_clip
        return (d_norm, ell), mu, lv

    @torch.no_grad()
    def infer_from_mu(self, x, a_diag_scaled, scale: float):
        mu, lv  = self.encode(x)
        s_d, t  = self._decode_heads(mu, L_out=x.shape[-1])
        d_norm  = a_diag_scaled * F.softplus(s_d) + self.eps
        ell     = torch.tanh(t)[..., :-1] * self.ell_clip
        d_un    = d_norm * scale
        return d_un, ell

# -------------------------
# LDLᵀ tridiagonal solves
# -------------------------

def ldlt_solve_tridiag(d: torch.Tensor, ell: torch.Tensor, rhs: torch.Tensor):
    """
    Solve (L D L^T) y = rhs with:
      D = diag(d) > 0 [B, L]
      L unit lower bidiagonal, subdiag ell [B, L-1]
    Shapes:
      d:   [B, L]   (requires_grad OK)
      ell: [B, L-1] (requires_grad OK)
      rhs: [B, L]
    Returns:
      y:   [B, L]
    Notes:
      - No in-place ops on views; uses scan (build lists + cat).
      - O(L) time and memory; differentiable w.r.t. d and ell.
    """
    B, L = rhs.shape
    assert d.shape == (B, L)
    assert ell.shape == (B, L-1) if L > 1 else ell.shape == (B, 0)

    # Forward solve: L w = rhs
    # w[:,0] = rhs[:,0]
    # w[:,i] = rhs[:,i] - ell[:,i-1] * w[:,i-1]
    w_cols = []
    w_prev = rhs[:, 0:1]
    w_cols.append(w_prev)
    for i in range(1, L):
        w_i = rhs[:, i:i+1] - ell[:, i-1:i] * w_prev
        w_cols.append(w_i)
        w_prev = w_i
    w = torch.cat(w_cols, dim=1)

    # Diagonal solve: D u = w
    u = w / d.clamp_min(1e-20)

    # Backward solve: L^T y = u  (superdiag = ell)
    # y[:,L-1] = u[:,L-1]
    # y[:,i]   = u[:,i] - ell[:,i] * y[:,i+1]
    y_cols_rev = []
    y_next = u[:, -1:]  # last column
    y_cols_rev.append(y_next)
    for i in range(L-2, -1, -1):
        y_i = u[:, i:i+1] - (ell[:, i:i+1] * y_next) if L > 1 else u[:, i:i+1]
        y_cols_rev.append(y_i)
        y_next = y_i
    y = torch.cat(list(reversed(y_cols_rev)), dim=1)

    return y


# -------------------------
# Loss & PCG eval routines
# -------------------------

def _probe_bank(L, probes, device, dtype, seed=123):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    v = torch.randint(0, 2, (probes, L), generator=g, dtype=torch.int8).float()
    v[v==0] = -1.0
    return v.to(device=device, dtype=dtype)

def residual_proxy_loss_ldlt(d_norm, ell, A_torch_csr, probes=3, seed=123):
    """
    E|| (I - M^{-1}(A/scale)) v ||^2 with Rademacher probes.
    M=L D L^T built from (d_norm, ell). Everything in normalized domain.
    """
    B, L = d_norm.shape
    V = _probe_bank(L, probes, d_norm.device, d_norm.dtype, seed=seed)  # [P,L]
    loss = 0.0
    for p in range(probes):
        v = V[p].expand(B, -1)               # [B,L]
        Av = torch.sparse.mm(A_torch_csr, v.T).T
        y  = ldlt_solve_tridiag(d_norm, ell, Av)  # M^{-1} Av
        r  = v - y
        loss += (r.pow(2).sum(dim=1)).mean()
    return loss / probes

def pcg_k_steps_loss_ldlt(A_t, d_norm, ell, probes=2, k=5, seed=123):
    """
    Minimize E[ ||r_k||^2 / ||r_0||^2 ] for k-step PCG on (A/scale)x=b,
    with preconditioner M = L D L^T (d_norm, ell).
    """
    B, L = d_norm.shape
    V = _probe_bank(L, probes, d_norm.device, d_norm.dtype, seed=seed)
    losses = []
    for p in range(probes):
        b = V[p].expand(B, -1)
        x = torch.zeros_like(b)
        r = b - torch.sparse.mm(A_t, x.T).T
        z = ldlt_solve_tridiag(d_norm, ell, r)   # z = M^{-1} r
        pvec = z.clone()
        r0 = (r.pow(2).sum(dim=1).clamp_min(1e-30)).sqrt()
        rz_old = (r*z).sum(dim=1).clamp_min(1e-30)
        for _ in range(k):
            Ap = torch.sparse.mm(A_t, pvec.T).T
            pAp = (pvec*Ap).sum(dim=1).clamp_min(1e-30)
            alpha = rz_old / pAp
            x = x + alpha[:,None]*pvec
            r = r - alpha[:,None]*Ap
            z = ldlt_solve_tridiag(d_norm, ell, r)
            rz_new = (r*z).sum(dim=1).clamp_min(1e-30)
            beta = rz_new / rz_old
            pvec = z + beta[:,None]*pvec
            rz_old = rz_new
        rk = r.norm(dim=1) / r0
        losses.append(rk.pow(2).mean())
    return torch.stack(losses).mean()

def ratio_guard_loss_diag(d_norm, a_diag_scaled, low=0.25, high=4.0):
    """Softly encourage d/diag(A) in [low, high] to keep scales sane."""
    r = d_norm / (a_diag_scaled.clamp_min(1e-20))
    return (F.relu(low - r) + F.relu(r - high)).mean()

def ell_l2_loss(ell):
    """Tiny L2 on subdiagonal to discourage huge coupling."""
    return (ell.pow(2)).mean()

# ---- Generic PCG with preconditioner callback (NumPy eval) ----

def pcg_5step_relres_with_precond(A: csr_matrix, b: np.ndarray, solve_M, steps: int = 5):
    """
    Generic k-step left PCG on Ax=b with SPD preconditioner M, given as solve_M(r).
    Returns ||r_k||/||r_0|| (no early stopping).
    """
    n = A.shape[0]
    x = np.zeros(n, dtype=np.float64)
    r = b.astype(np.float64, copy=False) - A @ x
    z = solve_M(r)
    p = z.copy()
    r0 = np.linalg.norm(r) + 1e-30
    rz_old = float(np.dot(r, z))
    if not np.isfinite(rz_old) or abs(rz_old) < 1e-300:
        return 1.0
    for _ in range(steps):
        Ap = A @ p
        pAp = float(np.dot(p, Ap)) + 1e-30
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = solve_M(r)
        rz_new = float(np.dot(r, z)) + 1e-30
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return float(np.linalg.norm(r) / r0)

def make_ldlt_numpy_solver(d_un: np.ndarray, ell_np: np.ndarray):
    """
    Build a numpy solve_M(r) for M = L D L^T with:
      d_un:  (L,) positive
      ell_np:(L-1,) in (-1,1)
    """
    d = d_un
    ell = ell_np
    def solve_M(r: np.ndarray) -> np.ndarray:
        r = r.astype(np.float64, copy=False)
        y = r.copy()
        # forward: L w = r
        for i in range(1, y.size):
            y[i] -= ell[i-1] * y[i-1]
        # diag: D u = w
        y /= np.clip(d, 1e-20, 1e20)
        # backward: L^T x = u
        for i in range(y.size-2, -1, -1):
            y[i] -= ell[i] * y[i+1]
        return y
    return solve_M

# --------------
# Train & eval
# --------------

def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    in_ch = 2*args.W + 1
    center_idx = args.W

    # Data
    train_ds = ShardedBandIterable(
        os.path.join(args.data_root, "train"),
        W=args.W, limit_per_shard=args.limit_per_shard,
        shuffle=True, seed=123, strict_spd_check=True,
        sym_tol=args.sym_tol, pd_tol=args.pd_tol, eigs_maxiter=args.eigs_maxiter,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, num_workers=args.workers,
        pin_memory=True, persistent_workers=(args.workers>0),
        collate_fn=lambda batch: batch[0],
    )
    ood_ds = ShardedBandIterable(
        os.path.join(args.data_root, "ood_test"),
        W=args.W, limit_per_shard=args.ood_limit_per_shard,
        shuffle=False, seed=123, strict_spd_check=True,
        sym_tol=args.sym_tol, pd_tol=args.pd_tol, eigs_maxiter=args.eigs_maxiter,
    )
    ood_loader = DataLoader(
        ood_ds, batch_size=1, num_workers=args.workers,
        pin_memory=True, persistent_workers=(args.workers>0),
        collate_fn=lambda batch: batch[0],
    )

    # Model
    model = BandVAE_LDLT(
        in_ch=in_ch, latent=args.latent, base=args.base, eps=1e-6,
        deterministic_latent=(not args.stochastic), ell_clip=args.ell_clip
    ).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Train ---
    ema = None
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            X = batch["X"].to(device=device, dtype=dtype)       # [C,L] normalized
            if X.dim() == 2: X = X.unsqueeze(0)                 # [B=1,C,L]
            a_diag_scaled = X[:, center_idx, :]                 # [B,L]
            scale = float(batch["scale"])
            Acsr = batch["Acsr"]

            # Build A/scale in torch CSR
            crow = torch.from_numpy(Acsr.indptr.astype(np.int64)).to(device)
            col  = torch.from_numpy(Acsr.indices.astype(np.int64)).to(device)
            val  = torch.from_numpy((Acsr.data/scale).astype(np.float32, copy=False)).to(device)
            Lsz  = Acsr.shape[0]
            A_t  = torch.sparse_csr_tensor(crow, col, val, size=(Lsz,Lsz), dtype=dtype, device=device)

            # Forward: (d_norm, ell)
            (d_norm, ell), mu, lv = model(X, a_diag_scaled=a_diag_scaled, scale=scale)
            d_norm = d_norm.clamp_min(1e-20)

            # Loss
            if args.loss == "proxy":
                loss_main = residual_proxy_loss_ldlt(d_norm, ell, A_t, probes=args.probes, seed=123)
            elif args.loss == "pcg":
                loss_main = pcg_k_steps_loss_ldlt(A_t, d_norm, ell, probes=args.probes, k=args.pcg_k, seed=123)
            else:
                raise ValueError("--loss must be 'proxy' or 'pcg'")

            loss = loss_main
            loss = loss + args.ratio_guard_w * ratio_guard_loss_diag(d_norm, a_diag_scaled)
            if args.ell_l2 > 0:
                loss = loss + args.ell_l2 * ell_l2_loss(ell)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            with torch.no_grad():
                cur = float(loss_main.detach().cpu())
                ema = cur if ema is None else (0.95*ema + 0.05*cur)
            pbar.set_postfix(loss=f"{cur:.4f}", ema=f"{ema:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"vae_tridiag_ldlt_W{args.W}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

    # --- OOD Eval ---
    model.eval()
    csv_path = os.path.join(args.out_dir, f"ood_eval_tridiag_W{args.W}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["index","S","W","rel_resid_5_id","rel_resid_5_jacobi","rel_resid_5_vae_tridiag"])
        idx = 0
        for batch in tqdm(ood_loader, desc="Evaluating OOD"):
            X = batch["X"].to(device=device, dtype=dtype)
            if X.dim() == 2: X = X.unsqueeze(0)
            a_diag_scaled = X[:, center_idx, :]
            scale = float(batch["scale"])
            S = int(batch["S"])
            Acsr = batch["Acsr"]

            with torch.no_grad():
                d_un, ell = model.infer_from_mu(X, a_diag_scaled=a_diag_scaled, scale=scale)  # [1,L], [1,L-1]
                d_np  = d_un.squeeze(0).detach().cpu().numpy()
                ell_np= ell.squeeze(0).detach().cpu().numpy()
                # Clamp for numerical safety
                d_np  = np.clip(d_np, 1e-20, 1e12)
                ell_np= np.clip(ell_np, -0.999, 0.999)

            D_jac = np.clip(Acsr.diagonal().copy(), 1e-20, 1e12)
            D_id  = np.ones(Acsr.shape[0], dtype=np.float64)

            rng = np.random.default_rng(1234)
            b = rng.normal(0, 1, size=Acsr.shape[0])
            b /= (np.linalg.norm(b) + 1e-12)

            # Identity
            r_id = pcg_5step_relres_with_precond(
                Acsr, b, solve_M=lambda r: r, steps=5
            )
            # Jacobi
            invD = 1.0 / np.clip(D_jac, 1e-20, 1e20)
            r_jac = pcg_5step_relres_with_precond(
                Acsr, b, solve_M=lambda r: invD * r, steps=5
            )
            # VAE (LDLᵀ tridiag)
            solve_M = make_ldlt_numpy_solver(d_np, ell_np)
            r_vae = pcg_5step_relres_with_precond(
                Acsr, b, solve_M=solve_M, steps=5
            )

            writer.writerow([idx, S, args.W, f"{r_id:.6e}", f"{r_jac:.6e}", f"{r_vae:.6e}"])
            idx += 1

    print(f"[CSV] OOD saved to {csv_path}")

    # Boxplot (log10 residual)
    try:
        df = pd.read_csv(csv_path)
        if len(df):
            fig, ax = plt.subplots(figsize=(6,4))
            idd = np.log10(np.maximum(df["rel_resid_5_id"].values.astype(float), 1e-16))
            jac = np.log10(np.maximum(df["rel_resid_5_jacobi"].values.astype(float), 1e-16))
            vae = np.log10(np.maximum(df["rel_resid_5_vae_tridiag"].values.astype(float), 1e-16))
            ax.boxplot([idd, jac, vae], labels=["Identity", "Jacobi", "VAE-Tridiag"])
            ax.set_ylabel("log10 residual after 5 PCG steps")
            ax.set_title(f"OOD: PCG-5 reduction (W={args.W})")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig_path = os.path.join(args.out_dir, f"ood_pcg5_box_W{args.W}.png")
            fig.savefig(fig_path, dpi=150)
            print(f"[Plot] Saved {fig_path}")
    except Exception as e:
        print(f"[Warn] Plotting failed: {e}")

    print("[Done] Training + OOD evaluation complete.")

# -----------
# CLI
# -----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_gen", help="Root with train/ and ood_test/ shards")
    ap.add_argument("--out_dir", type=str, default="runs_tridiag", help="Output directory")
    ap.add_argument("--W", type=int, default=5, help="Band half-width (offsets -W..+W)")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--loss", type=str, default="pcg", choices=["proxy","pcg"], help="Training objective")
    ap.add_argument("--probes", type=int, default=8, help="Rademacher probes")
    ap.add_argument("--pcg_k", type=int, default=5, help="Steps for pcg loss if --loss pcg")
    ap.add_argument("--ratio_guard_w", type=float, default=1e-3, help="Weight for d/diag(A) ratio guard")
    ap.add_argument("--ell_l2", type=float, default=1e-5, help="L2 penalty for subdiagonal ell")

    ap.add_argument("--stochastic", action="store_true", help="Use stochastic z (default deterministic)")
    ap.add_argument("--ell_clip", type=float, default=0.98, help="Max |ell| via tanh scaling to keep conditioning sane")

    # Dataset limits for quick runs / debugging
    ap.add_argument("--limit_per_shard", type=int, default=None, help="Train samples per shard")
    ap.add_argument("--ood_limit_per_shard", type=int, default=None, help="Eval samples per shard")

    # SPD checks
    ap.add_argument("--sym_tol", type=float, default=1e-8, help="Symmetry tolerance for ||A-A^T||_inf")
    ap.add_argument("--pd_tol", type=float, default=1e-12, help="Minimum λ_min threshold to accept SPD")
    ap.add_argument("--eigs_maxiter", type=int, default=200, help="Max iters for eigsh('SA') in SPD check")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
