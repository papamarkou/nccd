#!/bin/bash

scripts=(
  tune_lr.sh
  plot_lr.sh
)

for file in "${scripts[@]}"
do
    echo "Executing $file..."
    ./$file
done
