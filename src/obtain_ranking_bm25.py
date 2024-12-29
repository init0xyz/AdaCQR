import argparse
import json
import os

import pytrec_eval
from pyserini.search.lucene import LuceneSearcher
from shared_utils import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


def read_data(dataset, read_by="all", is_test=False):
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

    print(examples[:10])

    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--query_type', type=str, default="rewrite")
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--preprocessed_data_path', type=str, default=None)
    parser.add_argument('--pyserini_index_path', type=str, default=None)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--cand_num', type=int, default=32)
    parser.add_argument('--qrel_path', type=str, default="/home/init0xyz/InfoCQR/datasets/topiocqa/qrels_train.txt")
    parser.add_argument('--output_path', type=str, default="./test.jsonl")
    args = parser.parse_args()

    os.makedirs(args.preprocessed_data_path, exist_ok=True)


    if args.task == "qrecc":
        qrels = json.load(open(args.qrel_path))
        k_1 = 0.82
        b = 0.68
    else:
        qrels = json.load(open(args.qrel_path))
        k_1 = 0.9
        b = 0.4


    data = open(args.data_path, encoding="utf-8")
    raw_examples = read_data(data, args.query_type)
    print(f'Total number of queries: {len(raw_examples)}')

    searcher = LuceneSearcher(args.pyserini_index_path)
    searcher.set_bm25(k_1, b)

    qid_list = []
    query_list = []

    for _idx, line in enumerate(raw_examples):
        qid, q = line
        real_qid = qid.rsplit("_", 1)[0]

        no_rels = False
        if args.split == "test" or args.split == "dev":
            if list(qrels[real_qid].keys())[0] == '':
                no_rels = True
        if no_rels:
            continue

        qid_list.append(qid)
        query_list.append(q)

    hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=16)

    out_sfx = args.data_file.lstrip(args.split+"_").strip(".json")
    with open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.query_type}_{out_sfx}_bm25.trec"), "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid,
                                                i+1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')

    qrel_path = args.qrel_path

    qrels = json.load(open(qrel_path))
    # sqrels = dict(filter(lambda x: x[0] in qid_list, qrels.items()))
    sqrels = dict(filter(lambda x: x[1] != {"": 1}, qrels.items())) # QReCC: filtering missings
    qrels_ndcg = sqrels

    session_names = qrels.keys()

    with open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.query_type}_{out_sfx}_bm25.trec")) as f:
        run_data = f.readlines()

    runs = {}
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    all_scores = {}

    for item in session_names:
        copyed_qrels = {}
        copyed_runs = {}
        for i in range(args.cand_num):
            sample_id = f"{item}_{i}"
            if sample_id not in runs.keys():
                continue

            copyed_qrels[sample_id] = qrels[item]
            copyed_runs[sample_id] = runs[sample_id]

        if not copyed_qrels:
            continue

        evaluator = pytrec_eval.RelevanceEvaluator(copyed_qrels, {"recall.100", "recip_rank"})
        metrics = evaluator.evaluate(copyed_runs)
        for cur_metric in metrics.keys():
            all_scores[cur_metric] = metrics[cur_metric]['recip_rank']

        max_key = max(metrics, key=lambda k: metrics[k]['recip_rank'])
        best_runs = copyed_runs[max_key]

    data = open(args.data_path, encoding="utf-8")
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

