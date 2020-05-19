#!/bin/bash

source env_vars.sh

python plot_f1.py \
    --data_filename $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/validation_thres_metrics.csv \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename f1_plot.jpg \
    --verbose
