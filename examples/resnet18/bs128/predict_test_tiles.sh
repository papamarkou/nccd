#!/bin/bash

source env_vars.sh

python predict_tiles.py \
    --data_path $DATADIR/aisi2019/nuclear_canister_data \
    --test_dirname test \
    --model_path $OUTDIR/model \
    --model $RESNETNAME \
    --batch_size $BSVALUE \
    --ps $PS \
    --wd $WD \
    --output_path $OUTDIR \
    --output_filename test_tile_preds.csv \
    --verbose
