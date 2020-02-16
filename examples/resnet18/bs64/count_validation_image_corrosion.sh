#!/bin/bash

python count_validation_image_corrosion.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet18/bs64/validation_tile_preds.csv \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_filename validation_image_corrosion_counts.csv \
    --verbose
