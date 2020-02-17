#!/bin/bash

python tune_thres.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --test_dirname validation \
    --model_path $HOME/opt/output/nccd/examples/resnet18/bs64/model \
    --model resnet18 \
    --batch_size 64 \
    --thresholds 1 2 \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_optimal_thres_filename validation_optimal_thres.csv \
    --output_thres_metrics_filename validation_thres_metrics.csv \
    --verbose
