#!/bin/bash

python predict_tiles.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --test_dirname test \
    --model_path $HOME/opt/output/nccd/examples/resnet18/bs64/model \
    --model resnet18 \
    --batch_size 64 \
    --ps 0.1 \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_filename test_tile_preds.csv \
    --verbose
