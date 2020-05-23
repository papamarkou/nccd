#!/bin/bash

source env_vars.sh

python tune_thres.py \
    --data_path $DATADIR/aisi2019/nuclear_canister_data \
    --test_dirname validation \
    --model_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME/model \
    --model $RESNETNAME \
    --batch_size $BSVALUE \
    --ps $PS \
    --wd $WD \
    --thres 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 \
    --mixup \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_optimal_thres_filename validation_optimal_thres.csv \
    --output_thres_metrics_filename validation_thres_metrics.csv \
    --verbose
