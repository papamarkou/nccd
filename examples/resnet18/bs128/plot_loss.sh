#!/bin/bash

source env_vars.sh

python plot_loss.py \
    --data_filename $OUTDIR/all_losses.csv \
    --output_path $OUTDIR \
    --output_filename loss_plot.jpg \
    --verbose
