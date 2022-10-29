#!/bin/bash

#for lookback in 6 12 24 48 72 96 120 144 168 192 336 504 672 720
for lookback in 24 48 96 336
do
  for future in 1 12 24 96 192 336
  do
    if [ $future -gt 25 ]
    then
    python main.py \
    --lookback $lookback \
    --future $future \
    --gpu \
    --reduce_complexity
    else
    python main.py \
    --lookback $lookback \
    --future $future \
    --gpu
    fi
  done
done