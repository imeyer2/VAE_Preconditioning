#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae_jacobi_band_basic_spdcheck.py — NaN-safe, aligned & stabilized

Changes vs last version:
- Default strict SPD check ON (skip bad samples).
- If strict is OFF, auto-fix non-SPD by shifting A <- A + tau*I (training only).
- Finite checks: skip batch if any non-finite values arise (A/scale, M, loss).
- Retains domain alignment, Jacobi init, deterministic latent, ratio guard,
  and optional unrolled PCG-k training loss.

Recommended run:
  python vae_jacobi_band_basic_spdcheck.py \
    --data_root data_gen --out_dir runs_basic_spd --W 5 \
    --loss pcg --pcg_k 5 --probes 4 --epochs 5 \
    --lr 5e-4 --ratio_guard_w 1e-3 --mu_l2 1e-5
"""

import os, glob, math, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from scipy.sparse import csr_matrix, issparse
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
    A = csr_matrix((val, col, crow), shape=(n0, n1), dtype=np.float64)
    return A

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
    # If diag has NaN/Inf, fall back to finite median
    finite_diag = diag[np.isfinite(diag)]
    scale = np.median(np.abs(finite_diag)) if finite_diag.size else 1.0
    scale = float(scale) + 1e-8
    Xn = (X / scale).astype(X.dtype, copy=False)
    return Xn, scale

# -----------------
# SPD sanity checks
# -----------------

def _gershgorin_bounds(A: csr_matrix):
    diag = A.diagonal()
    absA = A.copy()
    absA.data = np.abs(absA.data)
    row_sums = np.array(absA.sum(axis=1)).ravel() - np.abs(diag)
    gmin = float(np.min(diag - row_sums))
    gmax = float(np.max(diag + row_sums))
    return gmin, gmax

def _eig_min(A: csr_matrix, maxiter=200, tol=1e-6):
    try:
        lmin = float(eigsh(A, k=1, which='SA', return_eigenvectors=False,
                           maxiter=maxiter, tol=tol)[0])
        return lmin
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

def add_diag_shift(A: csr_matrix, tau: float) -> csr_matrix:
    """Return A + tau I (CSR), tau>=0."""
    if tau <= 0: return A
    A = A.tolil(copy=True)
    A.setdiag(A.diagonal() + tau)
    return A.tocsr()

# -------------
# Dataset
# -------------

class ShardedBandIterable(IterableDataset):
    """
    Streams (X, scale, A, S) from shards with SPD checks and NaN-safety.
    Yields:
      X: torch.FloatTensor [C=2W+1, L=S^2] (normalized)
      scale: float
      Acsr: SciPy CSR (unscaled, float64)  [possibly shifted if strict=False]
      S: int
      meta: dict
    """
    def __init__(self, folder: str, W: int, limit_per_shard: int = None,
                 shuffle=False, seed=123, strict_spd_check=True,
                 sym_tol=1e-8, pd_tol=1e-12, eigs_maxiter=200):
        super().__init__()
        self.folder = folder
        self.W = W
        self.limit_per_shard = limit_per_shard
        self.shuffle = shuffle
        self.seed = seed
        self.strict = strict_spd_check
        self.sym_tol = sym_tol
        self.pd_tol = pd_tol
        self.eigs_maxiter = eigs_maxiter
        self.num_dropped = 0

    def __iter__(self):
        shard_paths = list_shards(self.folder)
        for spath in shard_paths:
            payload = load_shard(spath)
            idxs = list(range(len(payload)))
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(abs(hash((spath, self.seed))) % (2**31))
                idxs = torch.randperm(len(idxs), generator=g).tolist()
            if self.limit_per_shard is not None:
                idxs = idxs[:self.limit_per_shard]

            for i in idxs:
                d = payload[i]
                S = get_S_from_payload(d)
                A = payload_to_csr(d)

                # quick finite check on raw data
                if not np.isfinite(A.data).all():
                    print(f"[WARN] Non-finite entries in A; skipping (shard={os.path.basename(spath)} idx={i})")
                    self.num_dropped += 1
                    continue

                ok, info = is_spd(A, sym_tol=self.sym_tol, pd_tol=self.pd_tol,
                                  maxiter=self.eigs_maxiter)
                if not ok:
                    msg = (f"[SPD-FAIL] shard={os.path.basename(spath)} idx={i} "
                           f"sym_err={info['sym_err']:.3e} lmin={info['lmin']:.3e}")
                    if self.strict:
                        print(msg + " -> dropped")
                        self.num_dropped += 1
                        continue
                    else:
                        # Auto-fix for training only: shift by tau so that min eig ~ pd_tol
                        tau = max(0.0, (self.pd_tol - (info['lmin'] if np.isfinite(info['lmin']) else 0.0)) + 1e-8)
                        if tau > 0:
                            A = add_diag_shift(A, tau)
                            print(msg + f" -> shifted by tau={tau:.3e} for training")

                X = build_band_from_csr(A, S, self.W, dtype=np.float32)
                Xn, scale = normalize_band(X, self.W)

                # finite guard on normalized band
                if not np.isfinite(Xn).all() or not np.isfinite(scale):
                    print(f"[WARN] Non-finite band/scale; skipping (shard={os.path.basename(spath)} idx={i})")
                    self.num_dropped += 1
                    continue

                yield {
                    "X": torch.from_numpy(Xn),  # [C, L] float32 (normalized by "scale")
                    "scale": scale,
                    "Acsr": A,                  # UNnormalized SciPy CSR (possibly shifted if strict=False)
                    "S": S,
                    "meta": d.get("meta", {})
                }

# -------------
# VAE (1D Conv)
# -------------

def _inv_softplus_one():
    # softplus(b) = 1  => b = log(exp(1)-1)
    return math.log(math.e - 1.0)

class BandVAE(nn.Module):
    """
    Convolutional VAE head producing a diagonal preconditioner.
    forward(): returns m in normalized domain (A/scale).
    infer_from_mu(): returns unnormalized m for PCG on A.
    """
    def __init__(self, in_ch: int, latent: int = 64, base: int = 64, eps: float = 1e-6,
                 deterministic_latent: bool = True):
        super().__init__()
        self.eps = eps
        self.deterministic = deterministic_latent

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
        self.to_s = nn.Conv1d(base, 1, kernel_size=1)
        nn.init.zeros_(self.to_s.weight)
        nn.init.constant_(self.to_s.bias, _inv_softplus_one())  # start at Jacobi: softplus=1

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
        s = self.to_s(y).squeeze(1)  # [B, L]
        return s

    def forward(self, x, a_diag_scaled, scale: float):
        B, C, L = x.shape
        mu, lv = self.encode(x)
        z = mu if self.deterministic else self.reparam(mu, lv)
        s = self.decode_s(z, L_out=L)
        m_diag_norm = a_diag_scaled * F.softplus(s) + self.eps
        return m_diag_norm, mu, lv

    @torch.no_grad()
    def infer_from_mu(self, x, a_diag_scaled, scale: float):
        mu, lv = self.encode(x)
        s = self.decode_s(mu, L_out=x.shape[-1])
        m_diag_norm = a_diag_scaled * F.softplus(s) + self.eps
        return m_diag_norm * scale

# -------------------------
# Loss & PCG eval routines
# -------------------------

def _probe_bank(L, probes, device, dtype, seed=123):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    v = torch.randint(0, 2, (probes, L), generator=g, dtype=torch.int8).float()
    v[v==0] = -1.0
    return v.to(device=device, dtype=dtype)

def residual_proxy_loss(M_diag_norm, A_torch_csr, probes=3, seed=123):
    B, L = M_diag_norm.shape
    V = _probe_bank(L, probes, M_diag_norm.device, M_diag_norm.dtype, seed=seed)
    Minv = 1.0 / M_diag_norm.clamp_min(1e-12)
    loss = 0.0
    for p in range(probes):
        v = V[p].expand(B, -1)
        Av = torch.sparse.mm(A_torch_csr, v.T).T
        r = v - (Av * Minv)
        loss += (r.pow(2).sum(dim=1)).mean()
    return loss / probes

def pcg_k_steps_loss(A_t, m_diag_norm, probes=2, k=5, seed=123):
    B, L = m_diag_norm.shape
    V = _probe_bank(L, probes, m_diag_norm.device, m_diag_norm.dtype, seed=seed)
    Minv = 1.0 / m_diag_norm.clamp_min(1e-12)
    losses = []
    for p in range(probes):
        b = V[p].expand(B, -1)
        x = torch.zeros_like(b)
        r = b - torch.sparse.mm(A_t, x.T).T
        z = Minv * r
        pvec = z.clone()
        r0 = (r.pow(2).sum(dim=1).clamp_min(1e-30)).sqrt()
        rz_old = (r*z).sum(dim=1).clamp_min(1e-30)
        for _ in range(k):
            Ap = torch.sparse.mm(A_t, pvec.T).T
            pAp = (pvec*Ap).sum(dim=1).clamp_min(1e-30)
            alpha = rz_old / pAp
            x = x + alpha[:,None]*pvec
            r = r - alpha[:,None]*Ap
            z = Minv * r
            rz_new = (r*z).sum(dim=1).clamp_min(1e-30)
            beta = rz_new / rz_old
            pvec = z + beta[:,None]*pvec
            rz_old = rz_new
        rk = r.norm(dim=1) / r0
        losses.append(rk.pow(2).mean())
    return torch.stack(losses).mean()

def ratio_guard_loss(m_diag_norm, a_diag_scaled, low=0.25, high=4.0):
    r = m_diag_norm / (a_diag_scaled.clamp_min(1e-20))
    pen = F.relu(low - r) + F.relu(r - high)
    return pen.mean()

def pcg_5step_relres(A: csr_matrix, b: np.ndarray, D: np.ndarray, steps: int = 5):
    n = A.shape[0]
    D = np.clip(D, 1e-20, 1e12).astype(np.float64, copy=False)
    invD = 1.0 / D
    x = np.zeros(n, dtype=np.float64)
    r = b.astype(np.float64, copy=False) - A @ x
    z = invD * r
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
        z = invD * r
        rz_new = float(np.dot(r, z)) + 1e-30
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return float(np.linalg.norm(r) / r0)

# --------------
# Train & eval
# --------------

def train(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    in_ch = 2*args.W + 1
    center_idx = args.W  # center band is offset 0

    # Data (train)
    train_ds = ShardedBandIterable(
        os.path.join(args.data_root, "train"),
        W=args.W,
        limit_per_shard=args.limit_per_shard,
        shuffle=True,
        seed=123,
        strict_spd_check=not args.lenient,  # strict by default
        sym_tol=args.sym_tol,
        pd_tol=args.pd_tol,
        eigs_maxiter=args.eigs_maxiter,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers>0),
        collate_fn=lambda batch: batch[0],
    )

    # Data (OOD eval)
    ood_ds = ShardedBandIterable(
        os.path.join(args.data_root, "ood_test"),
        W=args.W,
        limit_per_shard=args.ood_limit_per_shard,
        shuffle=False,
        strict_spd_check=True,  # eval should be strict
        sym_tol=args.sym_tol,
        pd_tol=args.pd_tol,
        eigs_maxiter=args.eigs_maxiter,
    )
    ood_loader = DataLoader(
        ood_ds,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers>0),
        collate_fn=lambda batch: batch[0],
    )

    # Model
    model = BandVAE(
        in_ch=in_ch, latent=args.latent, base=args.base, eps=1e-6,
        deterministic_latent=(not args.stochastic)
    ).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Train ---
    ema = None
    skipped_batches = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # Build normalized inputs
            X = batch["X"].to(device=device, dtype=dtype)       # [C,L]
            if X.dim() == 2: X = X.unsqueeze(0)                 # [1,C,L]
            a_diag_scaled = X[:, center_idx, :]                 # [B,L]
            scale = float(batch["scale"])

            Acsr = batch["Acsr"]  # may be shifted if lenient
            # Build torch CSR in normalized domain (A/scale)
            crow = torch.from_numpy(Acsr.indptr.astype(np.int64))
            col  = torch.from_numpy(Acsr.indices.astype(np.int64))
            val_np = (Acsr.data / scale).astype(np.float32, copy=False)

            # Finite checks before creating tensor
            if not np.isfinite(val_np).all():
                print("[WARN] Non-finite A/scale values; skipping batch.")
                skipped_batches += 1
                continue

            crow = crow.to(device)
            col  = col.to(device)
            val  = torch.from_numpy(val_np).to(device)

            # Construct sparse CSR
            L = Acsr.shape[0]
            A_t = torch.sparse_csr_tensor(crow, col, val, size=(L,L), dtype=dtype, device=device)

            # Final finite guard on A values
            try:
                if not torch.isfinite(A_t.values()).all():
                    print("[WARN] Non-finite sparse values; skipping batch.")
                    skipped_batches += 1
                    continue
            except RuntimeError:
                # Some torch versions may not expose .values() for CSR; fall back to val tensor
                if not torch.isfinite(val).all():
                    print("[WARN] Non-finite sparse values; skipping batch.")
                    skipped_batches += 1
                    continue

            # Forward
            m_diag_norm, mu, lv = model(X, a_diag_scaled=a_diag_scaled, scale=scale)
            m_diag_norm = m_diag_norm.clamp_min(1e-20)

            # Finite guard on model output
            if not torch.isfinite(m_diag_norm).all():
                print("[WARN] Non-finite M output; skipping batch.")
                skipped_batches += 1
                continue

            # Loss selection
            if args.loss == "proxy":
                loss_main = residual_proxy_loss(m_diag_norm, A_t, probes=args.probes, seed=123)
            elif args.loss == "pcg":
                loss_main = pcg_k_steps_loss(A_t, m_diag_norm, probes=args.probes, k=args.pcg_k, seed=123)
            else:
                raise ValueError("--loss must be 'proxy' or 'pcg'")

            # Finite guard on loss_main
            if not torch.isfinite(loss_main):
                print("[WARN] Non-finite loss_main; skipping batch.")
                skipped_batches += 1
                continue

            # Small stabilizers
            loss = loss_main
            loss = loss + args.ratio_guard_w * ratio_guard_loss(m_diag_norm, a_diag_scaled)
            if args.mu_l2 > 0:
                loss = loss + args.mu_l2 * (mu.pow(2).mean())

            if not torch.isfinite(loss):
                print("[WARN] Non-finite total loss; skipping batch.")
                skipped_batches += 1
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            with torch.no_grad():
                cur = float(loss_main.detach().cpu()) if torch.isfinite(loss_main) else float('nan')
                ema = cur if (ema is None or not np.isfinite(ema)) else (0.95*ema + 0.05*cur)
            pbar.set_postfix(loss=f"{cur:.4f}", ema=f"{(ema if ema is not None else float('nan')):.4f}",
                             skipped=skipped_batches)

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"vae_band_basic_spd_W{args.W}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

    # --- OOD Eval ---
    model.eval()
    csv_path = os.path.join(args.out_dir, f"ood_eval_basic_spd_W{args.W}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["index","S","W","rel_resid_5_id","rel_resid_5_jacobi","rel_resid_5_vae"])
        idx = 0
        for batch in tqdm(ood_loader, desc="Evaluating OOD"):
            X = batch["X"].to(device=device, dtype=dtype)
            if X.dim() == 2: X = X.unsqueeze(0)
            a_diag_scaled = X[:, center_idx, :]
            scale = float(batch["scale"])
            S = int(batch["S"])
            Acsr = batch["Acsr"]

            with torch.no_grad():
                m_diag = model.infer_from_mu(X, a_diag_scaled=a_diag_scaled, scale=scale)
                m_np = m_diag.squeeze(0).detach().cpu().numpy()
                if not np.isfinite(m_np).all():
                    print("[WARN] Non-finite VAE diag at eval; using Jacobi for this sample.")
                    D_vae = np.clip(Acsr.diagonal().copy(), 1e-20, 1e12)
                else:
                    D_vae = np.clip(m_np, 1e-20, 1e12)

            D_jac = np.clip(Acsr.diagonal().copy(), 1e-20, 1e12)
            D_id  = np.ones(Acsr.shape[0], dtype=np.float64)

            rng = np.random.default_rng(1234)
            b = rng.normal(0, 1, size=Acsr.shape[0])
            b /= (np.linalg.norm(b) + 1e-12)

            r_id  = pcg_5step_relres(Acsr, b, D_id,  steps=5)
            r_jac = pcg_5step_relres(Acsr, b, D_jac, steps=5)
            r_vae = pcg_5step_relres(Acsr, b, D_vae, steps=5)

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
            vae = np.log10(np.maximum(df["rel_resid_5_vae"].values.astype(float), 1e-16))
            ax.boxplot([idd, jac, vae], labels=["Identity", "Jacobi", "VAE-Jacobi"])
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
    ap.add_argument("--out_dir", type=str, default="runs_basic_spd", help="Output directory")
    ap.add_argument("--W", type=int, default=5, help="Band half-width (offsets -W..+W)")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--loss", type=str, default="pcg", choices=["proxy","pcg"], help="Training objective")
    ap.add_argument("--probes", type=int, default=4, help="Rademacher probes")
    ap.add_argument("--pcg_k", type=int, default=5, help="Steps for pcg loss if --loss pcg")
    ap.add_argument("--ratio_guard_w", type=float, default=1e-3, help="Weight for ratio guard")
    ap.add_argument("--mu_l2", type=float, default=1e-5, help="Optional ||mu||^2 weight")
    ap.add_argument("--stochastic", action="store_true", help="Use stochastic z (default deterministic)")

    # Dataset limits for quick runs / debugging
    ap.add_argument("--limit_per_shard", type=int, default=None, help="Train samples per shard")
    ap.add_argument("--ood_limit_per_shard", type=int, default=None, help="Eval samples per shard")

    # SPD checks
    ap.add_argument("--lenient", action="store_true",
                    help="If set, do NOT drop non-SPD; instead shift A by tau*I for training (warnings shown).")
    ap.add_argument("--sym_tol", type=float, default=1e-8, help="Symmetry tolerance for ||A-A^T||_inf")
    ap.add_argument("--pd_tol", type=float, default=1e-12, help="Minimum λ_min threshold to accept SPD")
    ap.add_argument("--eigs_maxiter", type=int, default=200, help="Max iters for eigsh('SA') in SPD check")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
