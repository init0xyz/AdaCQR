import argparse
import json
import os
from glob import glob

from transformers import AutoTokenizer
from utils import get_logger, set_seed
from utils.indexing_utils import DocumentCollection

logger = get_logger(__name__)

"""
tokenize all passages
"""


def main(args):
    set_seed(args.random_seed)

    # ctx_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    ctx_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-roberta-base-ance-firstp')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    if not os.path.exists(os.path.join(args.output_path, args.task)):
        os.mkdir(os.path.join(args.output_path, args.task))

    if args.task == "qrecc":
        passage_files = glob(f"{args.test_collection_path}/collection-paragraph/*/*.jsonl")
    elif args.task == "topiocqa":
        passage_files = glob(f"{args.test_collection_path}/collections/*.jsonl")
    elif args.task == "cast":
        passage_files = glob(f"{args.test_collection_path}/collections/*.jsonl")
    elif args.task == "cast21":
        passage_files = glob(f"{args.test_collection_path}/collections/*.jsonl")


    logger.info(f"Overall {len(passage_files)} documents")

    if not os.path.exists(os.path.join(args.output_path, args.task, "collections")):
        os.mkdir(os.path.join(args.output_path, args.task, "collections"))

    output_path = os.path.join(args.output_path, args.task, "collections", "data.h5")
    pid_file_path = os.path.join(args.output_path, args.task, "collections", "pid2index.pkl")
    collection = DocumentCollection(output_path, max_passage_length=args.max_passage_length, pid_file_path=pid_file_path)
    collection.write_h5(passage_files, ctx_tokenizer)

    config = vars(args)
    with open(os.path.join(args.output_path, args.task, f"config_{args.suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--max_passage_length', type=int, default=384)
    parser.add_argument('--test_collection_path', type=str, default="")
    parser.add_argument('--pyserini_index_path', type=str, default=None)
    args = parser.parse_args()
    print(args)
    main(args)
