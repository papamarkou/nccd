#!/bin/bash

python train.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --test_dirname validation \
    --model_path $HOME/opt/output/nccd/examples/resnet18/bs64/model \
    --model resnet18 \
    --batch_size 64 \
    --thresholds 1 2 \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --verbose
