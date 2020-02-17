#!/bin/bash

python plot_lr.py \
    --data_filename $HOME/opt/output/nccd/examples/resnet18/bs64/tuning_lrs_and_losses.csv \
    --output_path $HOME/opt/output/nccd/examples/resnet18/bs64 \
    --output_filename lr_plot.jpg \
    --verbose
