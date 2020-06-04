#!/bin/bash

source env_vars.sh

python predict_test_images.py \
    --data_filename $OUTDIR/test_image_corrosion_counts.csv \
    --thres 11 \
    --output_path $OUTDIR \
    --output_filename test_image_preds.csv \
    --verbose
