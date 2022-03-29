sub_num=$1
CUDA_VISIBLE_DEVICES=$sub_num python ../fairseq/interactive.py ./corpus/enfr/ \
-s en -t fr \
--path ./checkpoints/enfr/model_avg.pt \
--beam 5 \
--nbest 5 \
--batch-size 128 \
--buffer-size 8000 \
--input ./th0.99_entail_dev_bpe.txt$sub_num > en-fr.log$sub_num \
