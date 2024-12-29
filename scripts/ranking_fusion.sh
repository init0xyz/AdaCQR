dense_candidates_path="/data1/laiyilong/AdaCQR-preview/output/topiocqa_train_datasets_stage2_dense.jsonl"
sparse_candidates_path="/data1/laiyilong/AdaCQR-preview/output/topiocqa_train_datasets_stage2_sparse.jsonl"
data_path="/data1/laiyilong/InfoCQR/datasets/topiocqa/train_with_gpt_rewrite.json"
output_path="/data1/laiyilong/AdaCQR-preview/output/topiocqa_train_datasets_stage2_fusioned.jsonl"

python src/ranking_fusion.py \
  --dense_candidates_path $dense_candidates_path \
  --sparse_candidates_path $sparse_candidates_path \
  --data_path $data_path \
  --output_path $output_path