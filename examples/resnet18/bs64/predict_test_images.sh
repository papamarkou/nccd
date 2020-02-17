#!/bin/bash

python predict_test_images.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet18/bs64/validation_image_corrosion_counts.csv \
    --thres 9 \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_filename test_image_preds.csv \
    --verbose
