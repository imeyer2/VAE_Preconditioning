#!/bin/bash



module load cuda/12.4

# python train_vae_band.py \
#   --data_root data_gen --out_dir runs_band_cond \
#   --W 5 --epochs 2 --device cuda --use_float32 \
#   --lr 5e-4 --probes 2 \
#   --w_residual 1.0 --w_spectral 0.5 --mean1_weight 1.0 \
#   --free_bits 0.05 --kl_warmup_steps 1000 --ood_limit_per_shard 10 \


# python train_vae_band.py \
#   --device cuda \
#   --data_root data_gen \
#   --out_dir runs_band_W5 \
#   --W 5 \
#   --epochs 5 \
#   --probes 3 \
#   --w_pos 0.1 --pos_mode softplus --pos_margin 1e-12




# python train2.py \
#   --data_root data_gen \
#   --out_dir runs_min \
#   --W 2 --epochs 5 --lr 5e-4 --beta 1.0 --use_float32


# python train_vae_band.py \
#   --data_root data_gen --out_dir runs_simple_W5 \
#   --W 5 --epochs 5 --device cuda --use_float32 --probes 2


# python simple.py \
#   --data_root data_gen \
#   --out_dir runs_band_basic \
#   --W 5 \
#   --epochs 5 \
#   --probes 3 \
#   --limit_per_shard 400 \
#   --ood_limit_per_shard 50


# python simple.py --data_root data_gen \
#   --out_dir runs_band_spd --W 5 --epochs 5


# python simple.py --loss pcg --pcg_k 5 --probes 4 --lr 5e-4 --ratio_guard_w 1e-3 --mu_l2 1e-5


# python vae_tridiag_ldlt.py \
#   --data_root data_gen --out_dir runs_tridiag --W 5 \
#   --loss pcg --pcg_k 5 --probes 8 --epochs 5 \
#   --lr 1e-3 --ratio_guard_w 1e-3 --ell_l2 1e-5

# python simpler.py \
#   --data_root data_gen --out_dir runs_min_hutch --W 5 \
#   --epochs 3 --lr 5e-4 --probes 8


python simplerer.py \
  --data_root data_gen --out_dir runs_min_hutch_v2 --W 5 \
  --epochs 5 --lr 5e-2 --probes 8