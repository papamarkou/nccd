#!/bin/bash

source env_vars.sh

python count_image_corrosion.py \
    --data_filename $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/test_tile_preds.csv \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename test_image_corrosion_counts.csv \
    --verbose
