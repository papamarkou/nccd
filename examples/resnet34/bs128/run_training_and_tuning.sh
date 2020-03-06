#!/bin/bash

scripts=(
  train.sh
  tune_thres.sh
)

for file in "${scripts[@]}"
do
    echo "Executing $file..."
    ./$file
done
