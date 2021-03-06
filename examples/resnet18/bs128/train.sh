#!/bin/bash

source env_vars.sh

python train.py \
    --data_path $DATADIR/aisi2019/nuclear_canister_data \
    --ds_tfms \
    --model $RESNETNAME \
    --batch_size $BSVALUE \
    --ps $PS \
    --wd $WD \
    --cyc_len 50 \
    --lr_interval \
    --lr_lower 0.000001 \
    --lr_upper 0.0005 \
    --output_path $OUTDIR \
    --output_model_filename model \
    --save_loss \
    --output_training_loss_filename training_losses.txt \
    --output_loss_filename all_losses.csv \
    --verbose
