#!/bin/bash

python predict_test_images.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet34/bs128/test_image_corrosion_counts.csv \
    --thres 13 \
    --output_path $HOME/opt/output/nccd/examples/resnet34/bs128 \
    --output_filename test_image_preds.csv \
    --verbose
