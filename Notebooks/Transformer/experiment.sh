#!/bin/bash


#for lookback in 6 12 24 48 72 96 120 144
#do
#  for future in 1 12 24 96 192 336 720
#  do
#    if [ $lookback -gt 48 ]
#    then
#        python main.py \
#        --model 'Autoformer' \
#        --seq_len $lookback \
#        --pred_len $future
#    else
#        python main.py \
#        --model 'Autoformer' \
#        --seq_len $lookback \
#        --label_len 5 \
#        --pred_len $future
#    fi
#  done
#  rm -r checkpoints
#done




#for val in True False
#do
#  for temp in True False
#  do
#    for pos in True False
#    do
#      for future in 1 12 48 96 192 336
#      do
#        python main.py \
#        --model 'Autoformer' \
#        --seq_len 96 \
#        --pred_len $future \
#        --positional_embedding $pos \
#        --value_embedding $val \
#        --temporal_embedding $temp
#        rm -r checkpoints
#      done
#    done
#  done
#done


for pos in False
do
  for val in False
  do
    for temp in False
    do
      if [ $pos == True ] && [ $val == False ] && [ $temp == False ]
      then
        echo "already done"
        coninue
      else
        for lookback in 6 72 96 120 144 168 192 336 504 672 720
        do
          for future in 1 12 24 96 192 336 720
          do
            if [ $lookback -gt 48 ]
            then
                python main.py \
                --model 'Transformer' \
                --seq_len $lookback \
                --pred_len $future \
                --positional_embedding $pos \
                --value_embedding $val \
                --temporal_embedding $temp
            else
                python main.py \
                --model 'Transformer' \
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
      fi
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



