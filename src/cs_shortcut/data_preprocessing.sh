#!/bin/bash

TASK=topiocqa
DATA_PATH=/data/laiyilong/InfoCQR/datasets
OUTPUT_PATH=/data/laiyilong/InfoCQR/datasets/preprocessed

# TASK=cast21
# DATA_PATH=/data1/laiyilong/infocqr/datasets
# OUTPUT_PATH=/data1/laiyilong/infocqr/datasets/preprocessed

python data_preprocessing.py \
  --task ${TASK} \
  --data_path ${DATA_PATH} \
  --output_path ${OUTPUT_PATH} \
  --max_passage_length 384 \
  --test_collection_path ${DATA_PATH}/${TASK}