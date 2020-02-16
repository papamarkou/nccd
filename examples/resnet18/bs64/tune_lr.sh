#!/bin/bash

python tune_lr.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --model resnet18 \
    --batch_size 64 \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_filename tuning_lrs_and_losses.csv \
    --verbose
