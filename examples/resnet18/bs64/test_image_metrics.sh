#!/bin/bash

python test_image_metrics.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet18/bs64/test_image_preds.csv \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_filename test_image_metrics.csv \
    --verbose
