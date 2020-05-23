#!/bin/bash

source env_vars.sh

python plot_lr.py \
    --data_filename $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/tuning_lrs_and_losses.csv \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename lr_plot.jpg \
    --verbose
