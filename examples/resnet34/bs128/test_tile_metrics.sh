#!/bin/bash

source env_vars.sh

python test_tile_metrics.py \
    --data_filename $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/test_tile_preds.csv \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename test_tile_metrics.csv \
    --verbose
