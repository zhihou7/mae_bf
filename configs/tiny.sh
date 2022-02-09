#!/usr/bin/env bash

source /usr/local/anaconda/4.2.0/etc/profile.d/conda.sh
conda activate pyt19

set -x
PY_ARGS=${@:1}


python main.py \
    --model deit_tiny_patch16_224 --batch-size 256 \
    --data-path /project/ZHIHOUDATA/Code/RIDE-LongTailRecognition/data/ImageNet_LT/ \
    --add_global 0 --output_dir ./tiny_output ${PY_ARGS}
