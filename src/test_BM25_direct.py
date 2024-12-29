import argparse
import json
import os
import time

import numpy as np
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

        if read_by == "rewrite":
            x = data["generated_query"]
        elif read_by == "gold":
            x = data["gold_rewrite"]
        elif read_by == "chatgpt":
            if data["chatgpt_rewrite"]:
                x = data["chatgpt_rewrite"].split("Bad Rewrite:")[0].strip()
            else:
                x = data["gold_rewrite"]
        else:
            raise Exception("Unsupported option!")

        examples.append([guid, x])

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
    parser.add_argument('--qrel_path', type=str, default="/home/init0xyz/InfoCQR/datasets/qrecc/qrels_test.txt")
    args = parser.parse_args()

    os.makedirs(args.preprocessed_data_path, exist_ok=True)


    if args.task == "qrecc":
        qrels = json.load(open(os.path.join("datasets/qrecc", f"qrels_{args.split}.txt")))
        k_1 = 0.82
        b = 0.68
    else:
        qrels = json.load(open(os.path.join("datasets/topiocqa", f"qrels_{args.split}.txt")))
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

        no_rels = False
        if args.split == "test" or args.split == "dev":
            if list(qrels[qid].keys())[0] == '':
                no_rels = True
        if no_rels:
            continue

        qid_list.append(qid)
        query_list.append(q)

    s_time = time.time()
    hits = searcher.batch_search(query_list, qid_list, k=args.top_k, threads=16)
    e_time = time.time()

    print(f"Time cost={e_time-s_time}")

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
    sqrels = dict(filter(lambda x: x[0] in qid_list, qrels.items()))
    sqrels = dict(filter(lambda x: x[1] != {"": 1}, sqrels.items())) # QReCC: filtering missings
    qrels_ndcg = sqrels

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

    evaluator = pytrec_eval.RelevanceEvaluator(sqrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)

    print(f"total valid queries={len(res.values())}")
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
    }

    print(res)

    # print(mrr_list)

    json.dump(runs, open("./results/BM25_retrieval_results.json", "w"), indent=4)

