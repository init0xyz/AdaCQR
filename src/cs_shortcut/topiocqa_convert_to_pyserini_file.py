import csv
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

id_col = 0
text_col= 1
title_col = 2

def main(wiki_file, output_file):
    with open(wiki_file) as input:
        reader = csv.reader(input, delimiter="\t")
        with open(output_file, 'w') as output:
            for i, row in enumerate(tqdm(reader)):
                if row[id_col] == "id":
                    continue
                title = row[title_col]
                text = row[text_col]
                title = ' '.join(title.split(' [SEP] '))
                obj = {"id": row[id_col], "contents": " ".join([title, text])}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wiki_file", type=str, default="/home/init0xyz/InfoCQR/datasets/topiocqa/full_wiki_segments.tsv")
    parser.add_argument("--output_file", type=str, default="/home/init0xyz/InfoCQR/datasets/topiocqa/full_wiki_segments_pyserini.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    main(args.wiki_file, args.output_file)
