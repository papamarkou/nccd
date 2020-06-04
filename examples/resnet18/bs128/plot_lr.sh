#!/bin/bash

source env_vars.sh

python plot_lr.py \
    --data_filename $OUTDIR/tuning_lrs_and_losses.csv \
    --output_path $OUTDIR \
    --output_filename lr_plot.jpg \
    --verbose
