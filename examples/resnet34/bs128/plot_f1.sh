#!/bin/bash

python plot_f1.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet34/bs128/validation_thres_metrics.csv \
    --output_path $HOME/opt/output/nccd/examples/resnet34/bs128 \
    --output_filename f1_plot.jpg \
    --verbose
