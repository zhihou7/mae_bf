#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt19

set -x
PY_ARGS=${@:1}


python main_pretrain.py \
    --add_global 74 \
    --insert_idx 8 \
    --drop_path_bt 0.6 \
    --add_norm_bt 0 \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /project/ZHIHOUDATA/Code/RIDE-LongTailRecognition/data/ImageNet_LT/  ${PY_ARGS}