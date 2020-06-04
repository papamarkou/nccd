#!/bin/bash

source env_vars.sh

python plot_f1.py \
    --data_filename $OUTDIR/validation_thres_metrics.csv \
    --output_path $OUTDIR \
    --output_filename f1_plot.jpg \
    --verbose
