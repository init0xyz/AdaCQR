#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# task="qrecc"
# preprocessed_data_path="/data/laiyilong/InfoCQR/output"
# index_path="/data/laiyilong/InfoCQR/datasets/preprocessed/qrecc/dense_index"

task="cast19"
preprocessed_data_path="/data1/laiyilong/infocqr/cast19_output"
dense_index_path="/data1/laiyilong/InfoCQR/datasets/preprocessed/cast/dense_index"
num_splits=10

# task="cast20"
# preprocessed_data_path="/data1/laiyilong/infocqr/cast20_output"
# dense_index_path="/data1/laiyilong/infocqr/datasets/preprocessed/cast/dense_index"
# num_splits=10


raw_data_path="/data1/laiyilong/InfoCQR/cast19_output/human_rewrite.json"

python run_dense_search.py --raw_data_path $raw_data_path --task $task --preprocessed_data_path $preprocessed_data_path --dense_index_path $dense_index_path --num_splits $num_splits
