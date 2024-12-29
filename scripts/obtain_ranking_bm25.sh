query_type="rewrite"
dataset="topiocqa"
if [ "$dataset" = "topiocqa" ]; then
    data_path="/home/init0xyz/AdaCQR-preview/topiocqa_candidates_test.jsonl"
    task="topiocqa"
    split="train"
    data_file="test.json"
    index_path="/home/init0xyz/InfoCQR/datasets/preprocessed/topiocqa/pyserini_index"
    preprocessed_data_path="/home/init0xyz/InfoCQR/topiocqa_output"
    qrel_path="/home/init0xyz/InfoCQR/datasets/topiocqa/qrels_train.txt"
    output_path="/home/init0xyz/InfoCQR/topiocqa_output/train_datasets_stage2_sparse.jsonl"
else
    task="qrecc"
    split="test"
    data_file="test.json"
    data_path="/home/init0xyz/AdaCQR/qrecc_LLM4CS_RAR.jsonl"
    index_path="/home/init0xyz/qrecc_bm25_index"
    preprocessed_data_path="/home/init0xyz/AdaCQR/qrecc_output"
    qrel_path="/home/init0xyz/AdaCQR/datasets/qrecc/qrels_test.txt"
    output_path="/home/init0xyz/AdaCQR/qrecc_output/train_datasets_stage2_sparse.jsonl"
fi

python src/obtain_ranking_bm25.py \
--data_path $data_path \
--preprocessed_data_path $preprocessed_data_path \
--pyserini_index_path $index_path \
--data_file $data_file \
--query_type $query_type \
--split $split \
--qrel_path $qrel_path \
--task $task \
