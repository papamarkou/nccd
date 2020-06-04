#!/bin/bash

source env_vars.sh

python count_image_corrosion.py \
    --data_filename $OUTDIR/test_tile_preds.csv \
    --output_path $OUTDIR \
    --output_filename test_image_corrosion_counts.csv \
    --verbose
