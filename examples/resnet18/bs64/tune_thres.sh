#!/bin/bash

python tune_thres.py \
    --data_path $HOME/opt/data/aisi2019/nuclear_canister_data \
    --test_dirname validation \
    --model_path $HOME/opt/output/nccd/examples/resnet18/bs64/model \
    --model resnet18 \
    --batch_size 64 \
    --ps 0.1 \
    --thres 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_optimal_thres_filename validation_optimal_thres.csv \
    --output_thres_metrics_filename validation_thres_metrics.csv \
    --verbose
