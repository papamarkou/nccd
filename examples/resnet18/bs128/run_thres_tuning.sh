#!/bin/bash

scripts=(
  tune_thres.sh
  plot_f1.sh
)

for file in "${scripts[@]}"
do
    echo "Executing $file..."
    ./$file
done
