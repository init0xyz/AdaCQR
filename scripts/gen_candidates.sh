export CUDA_VISIBLE_DEVICES=0

model_checkpoint_path="/data/laiyilong/InfoCQR/checkpoints/t5_rerank_rewrite_with_gpt_margin_0dot1_retrain_with_sprse_dense/epoch=0-step=3798.ckpt"
input_file_path="/data/laiyilong/InfoCQR/datasets/qrecc/train_candidates_with_sparse_dense_score.json"
output_file_path="/data/laiyilong/InfoCQR/output/qrecc_train_candidates_stage3.json"

python src/gen_candidates.py \
--input_file_path $input_file_path \
--checkpoint_path $model_checkpoint_path \
--out_file_path $output_file_path \
--model_name "google-t5/t5-base" \
--cand_num 32 \