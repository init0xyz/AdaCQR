query_type="rewrite"
dataset="topiocqa"
if [ "$dataset" = "topiocqa" ]; then
    data_path="/data1/laiyilong/AdaCQR-preview/topiocqa_candidates_test.jsonl"
    task="topiocqa"
    split="train"
    data_file="test.json"
    index_path="/data1/laiyilong/InfoCQR/datasets/topiocqa/dense_index"
    preprocessed_data_path="/data1/laiyilong/AdaCQR-preview/output"
    qrel_path="/data1/laiyilong/AdaRewriter/datasets/topiocqa_qrel.json"
    output_path="/data1/laiyilong/AdaCQR-preview/output/train_datasets_stage2_dense.jsonl"
    num_splits=8
else
    task="qrecc"
    split="test"
    data_file="test.json"
    data_path="/home/init0xyz/AdaCQR/qrecc.jsonl"
    index_path="/home/init0xyz/qrecc_bm25_index"
    preprocessed_data_path="/home/init0xyz/AdaCQR/qrecc_output"
    qrel_path="/home/init0xyz/AdaCQR/datasets/qrecc/qrels_test.txt"
    output_path="/home/init0xyz/AdaCQR/qrecc_output/train_datasets_stage2_dense.jsonl"
    num_splits=10
fi

python src/obtain_ranking_ance.py \
--raw_data_path $data_path \
--preprocessed_data_path $preprocessed_data_path \
--dense_index_path $index_path \
--num_splits $num_splits \
--data_file $data_file \
--query_type $query_type \
--split $split \
--qrel_path $qrel_path \
--output_path $output_path \
--task $task \
--cand_num 32 \
