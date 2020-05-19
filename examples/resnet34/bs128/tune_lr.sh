#!/bin/bash

source env_vars.sh

python tune_lr.py \
    --data_path $DATADIR/aisi2019/nuclear_canister_data \
    --model $RESNETNAME \
    --batch_size $BSVALUE \
    --ps 0.1 \
    --end_lr 0.5 \
    --num_lr_iters 200 \
    --output_path $OUTDIR/nccd/examples/$RESNETNAME/$BSNAME \
    --output_filename tuning_lrs_and_losses.csv \
    --verbose
