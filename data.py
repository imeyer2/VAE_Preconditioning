#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_sharded.py — Fast sharded data generator + DataLoader-ready readers for VAE OOD experiments.

Features
--------
1) Generates SPD systems A u = f for steady diffusion -div(a grad u) = f on [0,1]^2
   using a cell-centered finite-volume 5-point scheme with harmonic face averaging.
2) Axes of variation with SINGLE-AXIS OOD (others held ID), controlled by --axis_ood:
   - coeff, anisotropy, geom, bc, res, rhs
3) Writes FEW LARGE SHARDS instead of many tiny files (much faster I/O):
   data_gen/
     train/
       shard_00000.pt
       shard_00001.pt
       ...
     ood_test/
       shard_00000.pt
       ...
     train_metadata.jsonl
     ood_test_metadata.jsonl
   Each shard is a list[dict] with CSR pieces + f + meta.
4) Includes DataLoader-ready readers:
   - ShardedIterableDataset(folder, shuffle=True): streams samples; auto worker-sharding
   - ShardedMapDataset(folder): random-access by (shard_idx, local_idx) under the hood

Usage
-----
# Generate data (default: 10k train / 1k test, shard size 1000)
python make_dataset_sharded.py --axis_ood coeff

# Smaller set & float32 saves
python make_dataset_sharded.py --axis_ood rhs --train_n 2000 --test_n 400 --shard_size 500 --a_float32 --f_float32

# Training (example)
from make_dataset_sharded import ShardedIterableDataset, collate_sparse_stream
ds = ShardedIterableDataset("data_gen/train", shuffle=True, seed=123)
loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True,
                                     persistent_workers=True, collate_fn=collate_sparse_stream)
for batch in loader:
    A = batch["A"]; f = batch["f"]; meta = batch["meta"]
    ...

Notes
-----
- SPD guaranteed by positive a(x) + harmonic face averaging; Dirichlet and Robin(beta>=0).
- Anisotropy is axis-aligned (keep 5-point). For rotated anisotropy later, switch to 9-point.
- "geom" is implemented via low-a masks (perforations/checkerboard), not mesh changes.
"""

import os, json, math, argparse, glob
import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from tqdm import trange
import psutil, tracemalloc

# ----------------------------
# Basic I/O
# ----------------------------

# --- add near the other save/load helpers ---
class ShardWriter:
    def __init__(self, out_folder: str, shard_size: int):
        os.makedirs(out_folder, exist_ok=True)
        self.out_folder = out_folder
        self.shard_size = shard_size
        self.buf = []
        self.count = 0
        self.shard_idx = 0
        self.paths = []

    def add(self, sample):
        self.buf.append(sample)
        self.count += 1
        if len(self.buf) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buf:
            return
        path = os.path.join(self.out_folder, f"shard_{self.shard_idx:05d}.pt")
        torch.save(self.buf, path)
        self.paths.append(path)
        self.shard_idx += 1
        self.buf.clear()

    def close(self):
        self._flush()
        return self.paths



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def jsonl_append(path: str, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def rng_uniform(rng, lo, hi):
    return lo + (hi - lo) * rng.random()

# ----------------------------
# Random fields (FFT-based)
# ----------------------------
def make_grf(rng, N, length_scale, variance=1.0, band=None):
    wn = rng.normal(0, 1, (N, N))
    wx = np.fft.rfftfreq(N) * (2*np.pi*N)
    wy = np.fft.fftfreq(N)  * (2*np.pi*N)
    Kx = wx[None, :]
    Ky = wy[:, None]
    K  = np.sqrt(Kx**2 + Ky**2)
    S  = np.exp(-(K * length_scale)**2)
    if band is not None:
        kmin, kmax = band
        B = (K >= kmin) & (K <= kmax)
        S = S * B
    F  = np.fft.rfft2(wn)
    Ff = F * S
    field = np.fft.irfft2(Ff, s=(N, N))
    field = field / (field.std() + 1e-12) * math.sqrt(variance)
    return field

def make_rhs(rng, N, mode: str):
    if mode == "midband":   band = (8.0, 32.0)
    elif mode == "lowband": band = (0.0, 6.0)
    elif mode == "highband":band = (40.0, 200.0)
    else: raise ValueError("Unknown rhs mode")
    f = make_grf(rng, N, length_scale=0.18, variance=1.0, band=band)
    f = f - f.mean()
    f = f / (np.linalg.norm(f) + 1e-12) * N
    return f

# ----------------------------
# Geometry-like heterogeneity (via a(x))
# ----------------------------
def apply_perforations(a, kind: str, a_min=1e-4, rng=None):
    N = a.shape[0]
    if kind == "none":
        return a
    if kind == "perforated":
        num = max(3, N // 32)
        yy, xx = np.mgrid[0:N, 0:N]
        for _ in range(num):
            cx = rng.integers(N); cy = rng.integers(N)
            r  = rng.integers(max(2, N//32), max(3, N//16))
            mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
            a[mask] = np.minimum(a[mask], a_min)
        return a
    if kind == "checkerboard":
        tiles = 8
        t = max(1, N // tiles)
        for i in range(tiles):
            for j in range(tiles):
                if (i + j) % 2 == 0:
                    a[i*t:(i+1)*t, j*t:(j+1)*t] = np.minimum(
                        a[i*t:(i+1)*t, j*t:(j+1)*t], a_min
                    )
        return a
    raise ValueError("unknown perforation kind")

# ----------------------------
# Discretization (SPD assembly)
# ----------------------------
def assemble_spd_matrix(N, ax, ay, bc="dirichlet", robin_beta=0.0):
    h = 1.0 / N
    def idx(i, j): return i*N + j
    rows, cols, data = [], [], []
    def harm(a, b): return 2.0 * a * b / (a + b + 1e-12)

    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            on_left, on_right = (j == 0), (j == N-1)
            on_bottom, on_top = (i == 0), (i == N-1)

            if (on_left or on_right or on_bottom or on_top):
                if bc == "dirichlet" or (bc == "robin_top" and (on_left or on_right or on_bottom)):
                    rows.append(p); cols.append(p); data.append(1.0)
                    continue

            diag = 0.0
            if j > 0:
                ax_w = harm(ax[i, j], ax[i, j-1]); val = ax_w / (h*h)
                rows.append(p); cols.append(idx(i, j-1)); data.append(-val); diag += val
            if j < N-1:
                ax_e = harm(ax[i, j], ax[i, j+1]); val = ax_e / (h*h)
                rows.append(p); cols.append(idx(i, j+1)); data.append(-val); diag += val
            if i > 0:
                ay_s = harm(ay[i, j], ay[i-1, j]); val = ay_s / (h*h)
                rows.append(p); cols.append(idx(i-1, j)); data.append(-val); diag += val
            if i < N-1:
                ay_n = harm(ay[i, j], ay[i+1, j]); val = ay_n / (h*h)
                rows.append(p); cols.append(idx(i+1, j)); data.append(-val); diag += val

            if bc == "robin_top" and on_top:
                diag += (robin_beta)

            rows.append(p); cols.append(p); data.append(diag)

    A = coo_matrix((data, (rows, cols)), shape=(N*N, N*N)).tocsr()
    return A

# ----------------------------
# Axis sampling & realization
# ----------------------------
def sample_axes(rng, which_axis_ood: str, phase: str):
    cfg = {}
    # Axis 5: RES
    if which_axis_ood == "res" and phase == "ood_test":
        N = rng.choice([32, 256])
    else:
        N = rng.choice([64, 128])
    cfg["N"] = int(N)

    # Axis 1: COEFF_FIELD
    if which_axis_ood == "coeff" and phase == "ood_test":
        if rng.random() < 0.5:
            length_scale = rng_uniform(rng, 0.03, 0.08); logsigma = rng_uniform(rng, 0.0, 0.2)
        else:
            length_scale = rng_uniform(rng, 0.10, 0.15); logsigma = rng_uniform(rng, 1.2, 1.6)
    else:
        length_scale = rng_uniform(rng, 0.15, 0.25); logsigma = rng_uniform(rng, 0.3, 0.7)
    cfg["length_scale"] = float(length_scale); cfg["lognormal_sigma"] = float(logsigma)

    # Axis 2: ANISOTROPY
    if which_axis_ood == "anisotropy" and phase == "ood_test":
        alpha = rng_uniform(rng, 10.0, 50.0)
    else:
        alpha = rng_uniform(rng, 0.5, 2.0)
    cfg["anisotropy_alpha"] = float(alpha)

    # Axis 3: GEOM-ish
    if which_axis_ood == "geom" and phase == "ood_test":
        geom_kind = rng.choice(["perforated", "checkerboard"])
    else:
        geom_kind = rng.choice(["none", "none", "perforated"])
    cfg["geom_kind"] = str(geom_kind)

    # Axis 4: BC
    if which_axis_ood == "bc" and phase == "ood_test":
        bc_kind = "robin_top"; robin_beta = rng_uniform(rng, 5.0, 20.0)
    else:
        bc_kind = "dirichlet"; robin_beta = 0.0
    cfg["bc_kind"] = str(bc_kind); cfg["robin_beta"] = float(robin_beta)

    # Axis 6: RHS
    if which_axis_ood == "rhs" and phase == "ood_test":
        rhs_mode = rng.choice(["lowband", "highband"])
    else:
        rhs_mode = "midband"
    cfg["rhs_mode"] = str(rhs_mode)
    return cfg

def realize_sample(rng, cfg):
    N      = cfg["N"]
    ell    = cfg["length_scale"]
    logsg  = cfg["lognormal_sigma"]
    alpha  = cfg["anisotropy_alpha"]
    g  = make_grf(rng, N, length_scale=ell, variance=1.0)
    a0 = np.exp(logsg * g); a0 = np.clip(a0, 1e-3, 1e+3)
    a0 = apply_perforations(a0, cfg["geom_kind"], a_min=1e-4, rng=rng)
    ax = a0.copy(); ay = a0 * alpha
    A  = assemble_spd_matrix(N=N, ax=ax, ay=ay, bc=cfg["bc_kind"], robin_beta=cfg["robin_beta"])
    f  = make_rhs(rng, N, mode=cfg["rhs_mode"]).reshape(-1)
    meta = dict(cfg); meta.update({"a_min": float(a0.min()), "a_max": float(a0.max()),
                                   "a_mean": float(a0.mean()), "nnz": int(A.nnz),
                                   "shape": [int(A.shape[0]), int(A.shape[1])]})
    return A, f, meta

# ----------------------------
# Sharded save/load
# ----------------------------
def pack_sample(A, f, meta, a_float32: bool, f_float32: bool):
    return {
        "A_crow_indices": torch.from_numpy(A.indptr.astype(np.int64, copy=False)),
        "A_col_indices" : torch.from_numpy(A.indices.astype(np.int64, copy=False)),
        "A_values"      : torch.from_numpy(A.data.astype(np.float32 if a_float32 else np.float64, copy=False)),
        "A_shape"       : (A.shape[0], A.shape[1]),
        "f"             : torch.from_numpy(f.astype(np.float32 if f_float32 else np.float64, copy=False)),
        "meta"          : meta
    }

def save_shards(folder: str, samples: list, shard_size: int):
    """
    Save samples as a list-of-dicts per shard. Returns list of shard paths.
    """
    ensure_dir(folder)
    shard_paths = []
    num = len(samples)
    n_shards = (num + shard_size - 1) // shard_size
    for s in range(n_shards):
        lo = s * shard_size
        hi = min((s+1)*shard_size, num)
        payload = samples[lo:hi]
        path = os.path.join(folder, f"shard_{s:05d}.pt")
        torch.save(payload, path)
        shard_paths.append(path)
    return shard_paths

def list_shards(folder: str):
    return sorted(glob.glob(os.path.join(folder, "shard_*.pt")))

# ----------------------------
# Datasets (DataLoader-ready)
# ----------------------------
def _worker_shard(items):
    info = get_worker_info()
    if info is None: return items
    return items[info.id::info.num_workers]

class ShardedIterableDataset(IterableDataset):
    """
    Streams samples from shard files. Each worker gets a disjoint subset of shards.
    If shuffle=True, shard order is shuffled deterministically by seed; samples within
    a shard are iterated in stored order (you can shuffle inside too if desired).
    """
    def __init__(self, folder: str, shuffle: bool = False, seed: int = 123, shuffle_within_shard: bool = False):
        super().__init__()
        self.folder = folder
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_within_shard = shuffle_within_shard

    def __iter__(self):
        shards = list_shards(self.folder)
        if self.shuffle:
            g = torch.Generator(); g.manual_seed(self.seed)
            perm = torch.randperm(len(shards), generator=g).tolist()
            shards = [shards[i] for i in perm]
        shards = _worker_shard(shards)

        for spath in shards:
            payload = torch.load(spath, map_location="cpu")
            if self.shuffle_within_shard:
                g = torch.Generator(); g.manual_seed(abs(hash((spath, self.seed))) % (2**31))
                perm = torch.randperm(len(payload), generator=g).tolist()
                seq = [payload[i] for i in perm]
            else:
                seq = payload

            for d in seq:
                A = torch.sparse_csr_tensor(d["A_crow_indices"], d["A_col_indices"], d["A_values"],
                                            size=d["A_shape"])
                yield {"A": A, "f": d["f"], "meta": d["meta"]}

class ShardedMapDataset(Dataset):
    """
    Random-access view over shards. Keeps only shard index list; loads 1 shard on demand.
    Index space is flattened over shards.
    """
    def __init__(self, folder: str):
        self.folder = folder
        self.shards = list_shards(folder)
        self.shard_sizes = []
        self._prefix = [0]
        total = 0
        for spath in self.shards:
            n = len(torch.load(spath, map_location="cpu"))
            self.shard_sizes.append(n)
            total += n
            self._prefix.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def _locate(self, idx):
        # find shard s.t. prefix[s] <= idx < prefix[s+1]
        import bisect
        s = bisect.bisect_right(self._prefix, idx) - 1
        local = idx - self._prefix[s]
        return s, local

    def __getitem__(self, idx):
        s, local = self._locate(idx)
        payload = torch.load(self.shards[s], map_location="cpu")
        d = payload[local]
        A = torch.sparse_csr_tensor(d["A_crow_indices"], d["A_col_indices"], d["A_values"],
                                    size=d["A_shape"])
        return {"A": A, "f": d["f"], "meta": d["meta"]}

# Collate helpers
def collate_sparse_stream(batch):
    # Streaming: prefer batch_size=1. If >1, return lists (shapes may differ).
    if len(batch) == 1: return batch[0]
    return {
        "A": [b["A"] for b in batch],
        "f": torch.stack([b["f"] for b in batch]) if batch[0]["f"].ndim == 1 else [b["f"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }

def collate_sparse_map(batch):
    # Same as above — customize if you group by N to make real batches.
    return collate_sparse_stream(batch)

# ----------------------------
# Main (generate & shard)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data_gen")
    ap.add_argument("--train_n", type=int, default=100)
    ap.add_argument("--test_n", type=int, default=50)
    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--axis_ood", type=str, default="coeff",
                    choices=["coeff", "anisotropy", "geom", "bc", "res", "rhs"])
    ap.add_argument("--a_float32", action="store_true", help="Store A values as float32 (default float64).")
    ap.add_argument("--f_float32", action="store_true", help="Store f as float32 (default float64).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    root = args.out_dir
    train_dir = os.path.join(root, "train")
    test_dir  = os.path.join(root, "ood_test")
    ensure_dir(train_dir); ensure_dir(test_dir)

    train_meta_path = os.path.join(root, "train_metadata.jsonl")
    test_meta_path  = os.path.join(root, "ood_test_metadata.jsonl")
    open(train_meta_path, "w").close()
    open(test_meta_path, "w").close()

    tracemalloc.start()
    proc = psutil.Process(os.getpid())

    def rss_gb():
        return proc.memory_info().rss / (1024**3)

    # ---- TRAIN (all ID) ----
    train_sw = ShardWriter(train_dir, args.shard_size)
    pbar = trange(args.train_n, desc="Generating train (ID)")
    for k in pbar:
        cfg = sample_axes(rng, which_axis_ood=args.axis_ood, phase="train")
        A, f, meta = realize_sample(rng, cfg)
        packed = pack_sample(A, f, meta, a_float32=args.a_float32, f_float32=args.f_float32)
        train_sw.add(packed)

        # metadata stream to disk (tiny)
        meta_out = dict(meta); meta_out.update({"split":"train", "index":k})
        jsonl_append(train_meta_path, meta_out)

        # lightweight live RAM/peak report every 50 samples
        if (k % 50) == 0:
            _, peak = tracemalloc.get_traced_memory()
            pbar.set_postfix(rss=f"{rss_gb():.2f}GB", peak=f"{peak/1024/1024/1024:.2f}GB")
    train_shards = train_sw.close()

    # ---- OOD TEST (selected axis OOD) ----
    test_sw = ShardWriter(test_dir, args.shard_size)
    pbar = trange(args.test_n, desc=f"Generating OOD test ({args.axis_ood})")
    for k in pbar:
        cfg = sample_axes(rng, which_axis_ood=args.axis_ood, phase="ood_test")
        A, f, meta = realize_sample(rng, cfg)
        packed = pack_sample(A, f, meta, a_float32=args.a_float32, f_float32=args.f_float32)
        test_sw.add(packed)

        meta_out = dict(meta); meta_out.update({"split":"ood_test", "index":k})
        jsonl_append(test_meta_path, meta_out)

        if (k % 50) == 0:
            _, peak = tracemalloc.get_traced_memory()
            pbar.set_postfix(rss=f"{rss_gb():.2f}GB", peak=f"{peak/1024/1024/1024:.2f}GB")
    test_shards = test_sw.close()

    print(
        f"\nDone.\n- Train samples: {args.train_n} → {len(train_shards)} shard(s) in {train_dir}\n"
        f"- OOD ({args.axis_ood}) samples: {args.test_n} → {len(test_shards)} shard(s) in {test_dir}\n"
        f"- Metadata: {train_meta_path}, {test_meta_path}\n"
        f"- Dtypes: A={'float32' if args.a_float32 else 'float64'}, f={'float32' if args.f_float32 else 'float64'}\n"
        f"- Shard size: {args.shard_size}"
    )

if __name__ == "__main__":
    main()
