export CUDA_VISIBLE_DEVICES=2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

dataset="topiocqa"
output_type="rewrite"

if [ "$dataset" = "topiocqa" ]; then
    test_data_path="/data1/laiyilong/infocqr/datasets/topiocqa/test.json"
    model_name="google-t5/t5-base"
    checkpoint_path="/data1/laiyilong/infocqr/checkpoints/t5_base_topiocqa_rerank_stage2/epoch=3-step=22047.ckpt"
    bm25_index_path="/data1/laiyilong/infocqr/datasets/pyserini_index"
    output_file_path="/data1/laiyilong/infocqr/cast19_output/test_stage2_e3_s22047.json"
else
    model_name="google-t5/t5-base"
    test_data_path="/data/laiyilong/InfoCQR/datasets/qrecc/test.json"
    checkpoint_path="/data/laiyilong/InfoCQR/checkpoints/t5_base_rewrite_raw/epoch=8-step=17554.ckpt"
    bm25_index_path="/data/laiyilong/InfoCQR/datasets/preprocessed/qrecc/pyserini_index"
    output_file_path="/data/laiyilong/InfoCQR/qrecc_output/test_t5_base_raw_epoch8_step17554.json"
fi

python src/train.py \
--test_data_path $test_data_path \
--checkpoint_path $checkpoint_path \
--bm25_index_path $bm25_index_path \
--out_file_path $output_file_path \
--output_type $output_type \
--model_name $model_name \
--max_concat_length 768 \
--test \