export CUDA_VISIBLE_DEVICES=2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
output_type="rewrite"
warmup_ratio=0.1
dataset="topiocqa"

if [ "$dataset" = "topiocqa" ]; then
    export WANDB_PROJECT="topiocqa_rewrite"
    model_name="google-t5/t5-base"
    train_data_path="/data1/laiyilong/infocqr/datasets/topiocqa/train_with_gpt_rewrite.json"
    dev_data_path="/data1/laiyilong/infocqr/datasets/topiocqa/dev.json"
    bm25_index_path="/data1/laiyilong/infocqr/datasets/pyserini_index"
    default_root_dir="./checkpoints/t5-base-topiocqa-stage1"
    k1=0.9
    b=0.4
else
    export WANDB_PROJECT="qrecc_rewrite"
    model_name="google-t5/t5-base"
    train_data_path="/data/laiyilong/InfoCQR/datasets/qrecc/train.json"
    dev_data_path="/data/laiyilong/InfoCQR/datasets/qrecc/dev.json"
    bm25_index_path="/data/laiyilong/InfoCQR/datasets/preprocessed/qrecc/pyserini_index"
    default_root_dir="./checkpoints/t5_base_rewrite_raw"
    k1=0.82
    b=0.68
fi


python src/train.py \
--train_data_path $train_data_path \
--dev_data_path $dev_data_path \
--bm25_index_path $bm25_index_path \
--output_type $output_type \
--warmup_ratio $warmup_ratio \
--default_root_dir $default_root_dir \
--limit_val_batches 1.0 \
--bm25_k1 $k1 \
--bm25_b $b \
--model_name $model_name \
--max_concat_length 768 \
--use_gpt_as_gold \
--wandb \