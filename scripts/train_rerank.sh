export CUDA_VISIBLE_DEVICES=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

dataset="topiocqa"

if [ "$dataset" = "topiocqa" ]; then
    export WANDB_PROJECT="topiocqa_rewrite"
    model_name="google-t5/t5-base"
    train_data_path="/data1/laiyilong/infocqr/datasets/topiocqa/train_candidates_stage2.json"
    dev_data_path="/data1/laiyilong/infocqr/datasets/topiocqa/dev.json"
    checkpoint_path="/data1/laiyilong/infocqr/checkpoints/t5-base-topiocqa-stage1/epoch=5-step=15629.ckpt"
    default_root_dir="./checkpoints/t5_base_topiocqa_rerank_stage2"
    bm25_index_path="/data1/laiyilong/infocqr/datasets/pyserini_index"
    k1=0.9
    b=0.4
else
    export WANDB_PROJECT="qrecc_rewrite"
    model_name="google-t5/t5-base"
    train_data_path="/data/laiyilong/InfoCQR/ablation_study_datasets/train_candidates_stage6_cos.json"
    dev_data_path="/data/laiyilong/InfoCQR/datasets/qrecc/dev.json"
    checkpoint_path="/data/laiyilong/InfoCQR/checkpoints/qrecc_t5_base_stage1/epoch=0-step=1899.ckpt"
    default_root_dir="./checkpoints/qrecc_t5_base_rewrite_stage7"
    bm25_index_path="/data/laiyilong/InfoCQR/datasets/preprocessed/qrecc/pyserini_index"
    k1=0.82
    b=0.68
fi

python src/train_rerank.py \
--train_data_path $train_data_path \
--dev_data_path $dev_data_path \
--checkpoint_path $checkpoint_path \
--bm25_index_path $bm25_index_path \
--default_root_dir $default_root_dir \
--limit_val_batches 1.0 \
--val_check_interval 0.05 \
--gradient_accumulate 8 \
--cand_num 32 \
--bm25_k1 $k1 \
--bm25_b $b \
--model_name $model_name \
--use_gpt_as_gold \
--wandb \
# --max_concat_length 768