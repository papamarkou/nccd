#!/bin/bash

python train.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --model resnet18 \
    --batch_size 64 \
    --ps 0.1 \
    --cyc_len 50 \
    --lr_lower 0.000001 \
    --lr_upper 0.1 \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_model_filename model \
    --save_loss \
    --output_loss_filename training_losses.txt \
    --verbose
