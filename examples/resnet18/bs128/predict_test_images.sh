#!/bin/bash

source env_vars.sh

python predict_test_images.py \
    --data_filename $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/test_image_corrosion_counts.csv \
    --thres 15 \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename test_image_preds.csv \
    --verbose
