#!/bin/bash

source env_vars.sh

python test_image_metrics.py \
    --data_filename $OUTDIR/test_image_preds.csv \
    --output_path $OUTDIR \
    --output_filename test_image_metrics.csv \
    --verbose
