import argparse
import json
import os

import pytrec_eval
from cs_shortcut.utils import get_logger
from cs_shortcut.utils.indexing_utils import DenseIndexer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = get_logger(__name__)


def read_qrecc_data(dataset, read_by, is_test=False):
    examples = []
    for data in tqdm(dataset):
        data = json.loads(data)
        guid = data["sample_id"]

        all_candidates = data["candidates"]

        for i in range(len(all_candidates)):
            x = all_candidates[i]
            examples.append([f"{guid}_{i}", x])

        if is_test:
            logger.info(f"{guid}: {x}")
            if len(examples) == 10:
                break

    return examples


def merge_scores(scores_list, topk):
    results = {}
    for rr in scores_list:
        for k, v in rr.items():
            if k not in results:
                results[k] = list(v.items())
            else:
                results[k].extend(list(v.items()))

    new_results = {}
    for k, v in results.items():
        new_results[k] = {}
        vv = sorted(v, key=lambda x: -x[1])
        for i in range(topk):
            pid, ss = vv[i]
            new_results[k][pid] = ss

    return new_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--query_type', type=str, default="rewrite")
    parser.add_argument('--raw_data_path', type=str, default="/data/laiyilong/InfoCQR/output/test_inforgqr_rewrite_filtered.jsonl")
    parser.add_argument('--preprocessed_data_path', type=str, default="/data/laiyilong/InfoCQR/output/")
    parser.add_argument('--dense_index_path', type=str, default="/data/laiyilong/InfoCQR/datasets/preprocessed/qrecc/dense_index")
    parser.add_argument('--cache_folder', type=str, default="/data/laiyilong/InfoCQR/checkpoints")
    parser.add_argument('--data_file', type=str, default="test.json")
    parser.add_argument('--num_splits', type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--qrel_path', type=str, default=None)
    parser.add_argument("--cand_num", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="./test.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.preprocessed_data_path):
        os.makedirs(args.preprocessed_data_path, exist_ok=True)

    # read data
    data = open(args.raw_data_path, encoding="utf-8")
    raw_examples = read_qrecc_data(data, args.query_type)
    print(f'Total number of queries: {len(raw_examples)}')

    qids = []
    queries = []
    for _, line in enumerate(raw_examples):
        qid, q = line
        if q:
            qids.append(qid)
            queries.append(q)
    print(f'Number of valid queries: {len(queries)}')

    # query encoder
    model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-ance-firstp', cache_folder=args.cache_folder)
    model.max_seq_length = 512
    # model.max_seq_length = 128

    # query embeddings
    embeddings = model.encode(queries,
                              batch_size=args.batch_size,
                              show_progress_bar=True)

    all_scores_list = []
    out_sfx = args.data_file.lstrip(args.split+"_").strip(".json")


    for spt in range(args.num_splits):
        all_scores = {}
        for _, line in enumerate(raw_examples):
            qid, q = line
            if not q:
                all_scores[qid] = {}
                continue

        # load passage ids
        pids_path = os.path.join(args.dense_index_path, f"doc_ids_{spt}.json")
        pids = json.load(open(pids_path))
        logger.info(f"Load {len(pids)} pids from {pids_path}")

        # load faiss index
        index_path = os.path.join(args.dense_index_path, f"index_test_{spt}.faiss")
        logger.info(f"Load index from {index_path}")
        indexer = DenseIndexer(dim=768,logger=logger)
        indexer.load_index(index_path)
        logger.info("Index loading success!!")

        scores = indexer.retrieve(embeddings, qids, pids)

        all_scores.update(scores)

        logger.info("Dense search finished")

        all_scores_list.append(all_scores)


    merged_results = merge_scores(all_scores_list, 100)
    json.dump(
            merged_results,
            open(os.path.join(args.preprocessed_data_path, f"{args.task}_{args.split}_BoN_retrieval_result_dpr.json"), "w"),
            indent=4
        )

    qrels = json.load(open(args.qrel_path))

    session_names = qrels.keys()

    selected_best_runs = {}
    all_scores = {}

    for item in session_names:
        copyed_qrels = {}
        copyed_runs = {}
        for i in range(args.cand_num):
            sample_id = f"{item}_{i}"
            if sample_id not in merged_results.keys():
                continue

            try:
                copyed_qrels[sample_id] = qrels[item]
                copyed_runs[sample_id] = merged_results[sample_id]
            except Exception:
                break

        evaluator = pytrec_eval.RelevanceEvaluator(copyed_qrels, {"recall.100", "recip_rank"})
        metrics = evaluator.evaluate(copyed_runs)
        for cur_metric in metrics.keys():
            all_scores[cur_metric] = metrics[cur_metric]['recip_rank']

    data = open(args.raw_data_path, encoding="utf-8")
    out_data = open(args.output_path, "w", encoding="utf-8")

    for _idx, line in enumerate(data):
        data = json.loads(line)
        guid = data["sample_id"]
        all_candidates = data["candidates"]
        candidates_result = []
        for i in range(len(all_candidates)):
            x = all_candidates[i]
            candidates_result.append([x, all_scores[f"{guid}_{i}"]])

        out_data.write(json.dumps({"sample_id": guid, "candidates": candidates_result}) + "\n")
