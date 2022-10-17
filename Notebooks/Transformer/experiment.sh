#!/bin/bash


for model in Transformer Autoformer Informer
do
  for pos in False True
  do
    for val in False True
    do
      for temp in False True
      do
        for lookback in 6 12 24 48 72 96 120 144 168 192 336 504 672 720
        do
          for future in 1 12 24 96 192 336 720
          do
            if [ $lookback -gt 48 ]
            then
                python main.py \
                --model $model \
                --seq_len $lookback \
                --pred_len $future \
                --positional_embedding $pos \
                --value_embedding $val \
                --temporal_embedding $temp
            else
                python main.py \
                --model $model \
                --seq_len $lookback \
                --label_len 5 \
                --pred_len $future \
                --positional_embedding $pos \
                --value_embedding $val \
                --temporal_embedding $temp
            fi
            rm -r checkpoints
          done
        done
      done
    done
  done
done

#for lookback in 6 12 24 48 72 96 120 144 168 192 336 504 672 720
#do
#  for future in 1 12 24 96 192 336 720
#  do
#    if [ $lookback -gt 48 ]
#    then
#        python main.py --seq_len $lookback --pred_len $future
#    else
#        python main.py --seq_len $lookback --label_len 5 --pred_len $future
#    fi
#  done
#done



