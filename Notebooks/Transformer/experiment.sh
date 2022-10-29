#!/bin/bash


for model in Transformer Informer Autoformer
do
  for decoder_len in 5 15 25 35 44
  do
        python main.py \
        --model $model \
        --seq_len 48 \
        --pred_len 12 \
        --label_len $decoder_len
        rm -r checkpoints
  done
done


#for lookback in 6 12 24 48 72 96 120 144 168 192 336 504 672
#for lookback in 24 48 96 336 720
#do
#  for future in 1 12 24 96 192 336 720
#  do
#    for model in Transformer Informer
#    do
#      for pos in True
#      do
#        for val in False
#        do
#          for temp in True
#          do
#	          if [ $pos = "False" ] && [ $val = "False" ] && [ $temp = "False" ]
#	          then
#	            echo "skipping loop because no embedding"
#	            continue
#	          fi
#            if [ $lookback -gt 48 ]
#            then
#                python main.py \
#                --model $model \
#                --seq_len $lookback \
#                --pred_len $future \
#                --positional_embedding $pos \
#                --value_embedding $val \
#                --temporal_embedding $temp
#            else
#                python main.py \
#                --model $model \
#                --seq_len $lookback \
#                --label_len 5 \
#                --pred_len $future \
#                --positional_embedding $pos \
#                --value_embedding $val \
#                --temporal_embedding $temp
#            fi
#            rm -r checkpoints
#          done
#        done
#      done
#    done
#  done
#done

#for lookback in 6 12 24 48 72 96 120 144 168 192 336 504 672
#do
#  for future in 1 12 24 96 192 336 720
#  do
#    for model in Transformer Informer Autoformer
#    do
#      for pos in True
#      do
#        for val in False
#        do
#          for temp in False
#          do
#	          if [ $pos = "False" ] && [ $val = "False" ] && [ $temp = "False" ]
#	          then
#	            echo "skipping loop because no embedding"
#	            continue
#	          fi
##	          if [ $pos = "True" ] && [ $val = "False" ] && [ $temp = "False" ]
##	          then
##	            echo "To be done for just pos embedding"
##	            continue
##	          fi
#            if [ $lookback -gt 48 ]
#            then
#                python main.py \
#                --model $model \
#                --seq_len $lookback \
#                --pred_len $future \
#                --positional_embedding $pos \
#                --value_embedding $val \
#                --temporal_embedding $temp
#            else
#                python main.py \
#                --model $model \
#                --seq_len $lookback \
#                --label_len 5 \
#                --pred_len $future \
#                --positional_embedding $pos \
#                --value_embedding $val \
#                --temporal_embedding $temp
#            fi
#            rm -r checkpoints
#          done
#        done
#      done
#    done
#  done
#done


#
#for lookback in 6 12 24 48 72 96 120 144 168 192 336 504 672 720
#do
#  for future in 1 12 24 96 192 336 720
#  do
#    for model in Transformer Informer Autoformer
#    do
#      for pos in False True
#      do
#        for val in False True
#        do
#          for temp in False True
#          do
#	          # shellcheck disable=SC1073
#	          if [ $pos = "False" ] && [ $val = "False" ] && [ $val = "False" ]
#	          then
#	            echo "skipping loop because no embedding"
#	            continue
#            else
#              if [ $lookback -gt 48 ]
#              then
#                  python main.py \
#                  --model $model \
#                  --seq_len $lookback \
#                  --pred_len $future \
#                  --positional_embedding $pos \
#                  --value_embedding $val \
#                  --temporal_embedding $temp
#              else
#                  python main.py \
#                  --model $model \
#                  --seq_len $lookback \
#                  --label_len 5 \
#                  --pred_len $future \
#                  --positional_embedding $pos \
#                  --value_embedding $val \
#                  --temporal_embedding $temp
#              fi
#              rm -r checkpoints
#            fi
#          done
#        done
#      done
#    done
#  done
#done
