#!/bin/bash

#for lookback in 6 12 24 48 72 96 120 144 168 192 336 504 672 720
for lookback in 720 672 504 336 192 168 144 120 96 72 48 24 12 6
do
  for future in 1 12 24 96 192 336 720
  do
    python main.py \
    --lookback $lookback \
    --future $future \
    --gpu
#    if [ $lookback -gt 168 ]
#    then
#      python main.py \
#      --lookback $lookback \
#      --future $future \
#      --gpu
#    else
#      python main.py \
#      --lookback $lookback \
#      --future $future
#    fi
  done
done