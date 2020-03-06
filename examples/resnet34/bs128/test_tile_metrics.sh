#!/bin/bash

python test_tile_metrics.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet34/bs128/test_tile_preds.csv \
    --output_path $HOME/opt/output/nccd/examples/resnet34/bs128 \
    --output_filename test_tile_metrics.csv \
    --verbose
