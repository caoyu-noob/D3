sub_num=$1
CUDA_VISIBLE_DEVICES=$sub_num python ../fairseq/interactive.py ./corpus/fren/ \
-s fr -t en \
--path ./checkpoints/fren/model_avg.pt \
--beam 5 \
--nbest 5 \
--batch-size 128 \
--buffer-size 8000 \
--input ./en-fr.out$sub_num > fr-en.log$sub_num \
