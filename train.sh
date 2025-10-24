#!/bin/bash



module load cuda/12.4

python train_vae_band.py \
  --data_root data_gen --out_dir runs_band_cond \
  --W 5 --epochs 10 --device cuda --use_float32 \
  --lr 5e-4 --probes 2 \
  --w_residual 1.0 --w_spectral 0.5 --mean1_weight 1.0 \
  --free_bits 0.05 --kl_warmup_steps 1000 --ood_limit_per_shard 10 \


# python train2.py \
#   --data_root data_gen \
#   --out_dir runs_min \
#   --W 2 --epochs 5 --lr 5e-4 --beta 1.0 --use_float32


# python train_vae_band.py \
#   --data_root data_gen --out_dir runs_simple_W5 \
#   --W 5 --epochs 5 --device cuda --use_float32 --probes 2
