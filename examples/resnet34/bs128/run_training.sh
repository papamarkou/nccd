#!/bin/bash

scripts=(
  train.sh
  plot_loss.sh
)

for file in "${scripts[@]}"
do
    echo "Executing $file..."
    ./$file
done
