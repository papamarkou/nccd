#!/bin/bash

source env_vars.sh

python test_image_metrics.py \
    --data_filename $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/test_image_preds.csv \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename test_image_metrics.csv \
    --verbose
