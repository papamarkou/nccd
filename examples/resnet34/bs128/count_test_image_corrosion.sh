#!/bin/bash

python count_image_corrosion.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet34/bs128/test_tile_preds.csv \
    --output_path $HOME/opt/output/nccd/examples/resnet34/bs128 \
    --output_filename test_image_corrosion_counts.csv \
    --verbose
