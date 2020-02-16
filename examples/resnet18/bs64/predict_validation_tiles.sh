#!/bin/bash

python predict_validation_tiles.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --test_dirname validation \
    --model_path $HOME/opt/output/nccd/examples/resnet18/bs64/model \
    --model resnet18 \
    --batch_size 64 \
    --output_path $HOME/opt/output \
    --output_filename validation_tile_preds.csv \
    --verbose
