#!/bin/bash

python train.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --model resnet34 \
    --batch_size 128 \
    --ps 0.1 \
    --cyc_len 20 \
    --lr_lower 0.000001 \
    --lr_upper 0.1 \
    --output_path $HOME/opt/output/nccd/examples/resnet34/bs128 \
    --output_model_filename model \
    --save_loss \
    --output_loss_filename training_losses.txt \
    --verbose
