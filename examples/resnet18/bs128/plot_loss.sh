#!/bin/bash

source env_vars.sh

python plot_loss.py \
    --data_filename $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/all_losses.csv \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename loss_plot.jpg \
    --verbose
