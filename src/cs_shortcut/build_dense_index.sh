export CUDA_VISIBLE_DEVICES=0,3
data_path="/data1/laiyilong/infocqr/datasets/preprocessed/"
task="topiocqa"
output_path="/data1/laiyilong/infocqr/datasets/preprocessed/topiocqa/dense_index"
num_splits=10

python build_dense_index_multi_gpus.py --data_path $data_path --task $task --output_path $output_path --num_splits $num_splits