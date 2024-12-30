export CUDA_VISIBLE_DEVICES=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
query_type="rewrite"

dataset="topiocqa"
if [ "$dataset" = "topiocqa" ]; then
    test_data_path="/data1/laiyilong/infocqr/datasets/topiocqa/test.json"
    model_name="google-t5/t5-base"
    checkpoint_path="your_checkpoint_path"
    bm25_index_path="/data1/laiyilong/infocqr/datasets/pyserini_index"
    output_file_path="/data1/laiyilong/infocqr/topiocqa_output/topiocqa_test.json"
elif [ "$dataset" = "qrecc" ]; then
    test_data_path="/data/laiyilong/InfoCQR/datasets/qrecc/test.json"
    model_name="google-t5/t5-base"
    checkpoint_path="your_checkpoint_path"
    bm25_index_path="/data/laiyilong/InfoCQR/datasets/preprocessed/qrecc/pyserini_index"
    output_file_path="/data/laiyilong/InfoCQR/qrecc_output/qrecc_test_only_dense_stage3.json"
else
    test_data_path="/data1/laiyilong/infocqr/datasets/cast/cast19_test.json"
    model_name="google-t5/t5-base"
    checkpoint_path="your_checkpoint_path"
    bm25_index_path="/data1/laiyilong/infocqr/datasets/pyserini_index"
    output_file_path="/data1/laiyilong/infocqr/cast19_output/test_stage2.json"
fi

python src/train_rerank.py \
--test_data_path $test_data_path \
--checkpoint_path $checkpoint_path \
--bm25_index_path $bm25_index_path \
--out_file_path $output_file_path \
--output_type $query_type \
--model_name $model_name \
--max_concat_length 768 \
--test \