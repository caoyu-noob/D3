python ../fairseq/scripts/average_checkpoints.py \
--input checkpoints/enfr/ \
--num-epoch-checkpoints 5 \
--output ./checkpoints/enfr/model_avg.pt \
