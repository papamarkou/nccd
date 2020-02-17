#!/bin/bash

scripts=(
  train.sh
  predict_validation_tiles.sh
  count_validation_image_corrosion.sh
  tune_thres.sh
)

for file in "${scripts[@]}"
do
    echo "Executing $file..."
    ./$file
done
