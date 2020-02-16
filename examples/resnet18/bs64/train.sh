#!/bin/bash

python trainer.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --model resnet18 \
    --batch_size 64 \
    --cyc_len 10 \
    --lr_lower 0.0001 \
    --lr_upper 0.1 \
    --output_path $HOME/workspace/software/source/nuclear_paper_code/output/resnet18/bs64 \
    --output_model_filename resnet18_bs64_model \
    --save_loss \
    --output_loss_filename resnet18_bs64_training_losses.txt \
    --verbose
