ROOT=./
SRC=en
TGT=fr
DICT=./

python ../fairseq/preprocess.py \
--source-lang en \
--target-lang fr \
--trainpref train \
--validpref valid \
--testpref test \
--destdir ../corpus/enfr \
--thresholdtgt 0 \
--thresholdsrc 0 \
--workers 90