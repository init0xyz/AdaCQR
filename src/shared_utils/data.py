import argparse
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def pad_seq_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[-max_length:]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids

    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length

    return input_ids, attention_mask


class RewriterDataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, file_name, training=True, use_prefix=True, use_data_percent=1.0, use_gpt_as_gold=False) -> None:
        self.examples = []
        self.training = training

        with open(file_name, encoding='utf-8') as f:
            data = f.readlines()

        n = len(data)
        n = int(use_data_percent * n)
        if n < len(data):
            random.seed(42)
            data = random.sample(data, n)

        for line in tqdm(data):
            input_ids = []
            record = json.loads(line)
            sample_id = record['sample_id']
            current_query = record['current_query']
            context = record['context']
            gold_rewrite = record['gold_rewrite']
            gold_answer = record['gold_answer']
            pos_docs_pids = record['pos_docs_pids']
            pos_docs_text = record['pos_docs_text']

            if use_prefix:
                current_query = "question:" + current_query
                first_context = True

            tokenized_query = tokenizer.encode(current_query, add_special_tokens = True, max_length = args.max_query_length)

            input_ids.extend(tokenized_query)
            for j in range(len(context) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length

                if use_prefix and first_context:
                    context[j] = "context: " + context[j]
                    first_context = False

                utt = tokenizer.encode(context[j], add_special_tokens=True, max_length= max_length, truncation=True)
                if len(input_ids) + len(utt) > args.max_concat_length:
                    input_ids += utt[:args.max_concat_length - len(input_ids) - 1] + [utt[-1]] # must ended with [SEP]
                    break
                else:
                    input_ids.extend(utt)

            input_ids, input_ids_mask = padding_seq_to_same_length(input_ids, max_pad_length=args.max_concat_length)

            if self.training:
                if args.output_type == "rewrite":
                    if use_gpt_as_gold:
                        target_seq = record["chatgpt_rewrite"]
                    else:
                        target_seq = gold_rewrite
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)
                elif args.output_type == "answer":
                    target_seq = gold_answer
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()

                for idx in range(len(pos_docs_pids)):
                    pos_docs = []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)

                    self.examples.append([sample_id,
                                        input_ids,
                                        input_ids_mask,
                                        labels,
                                        target_seq,
                                        pos_docs,
                                        pos_docs_mask
                                        ])

                if len(pos_docs_pids) == 0:
                    self.examples.append([sample_id,
                                          input_ids,
                                          input_ids_mask,
                                          labels,
                                          target_seq,
                                          [0] * args.max_doc_length,
                                          [0] * args.max_doc_length
                                          ])

            else:
                if args.output_type == "rewrite":
                    target_seq = gold_rewrite
                else:
                    target_seq = gold_answer

                if len(pos_docs_pids) != 0:
                    self.examples.append([sample_id,
                                        input_ids,
                                        input_ids_mask,
                                        target_seq,
                                        pos_docs_pids
                                        ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch: list):
            collated_dict = {
                "sample_ids": [],
                "input_ids": [],
                "input_masks": [],
                "label_ids": [],
                "pos_docs_ids": [],
                "pos_docs_masks": [],
                "labels": []
            }

            for example in batch:
                if len(example) == 5:
                    collated_dict["sample_ids"].append(example[0])
                    collated_dict["input_ids"].append(example[1])
                    collated_dict["input_masks"].append(example[2])
                    collated_dict["labels"].append(example[3])
                    collated_dict["pos_docs_ids"].append(example[4])
                else:
                    collated_dict["sample_ids"].append(example[0])
                    collated_dict["input_ids"].append(example[1])
                    collated_dict["input_masks"].append(example[2])
                    collated_dict["label_ids"].append(example[3])
                    collated_dict["labels"].append(example[4])
                    collated_dict["pos_docs_ids"].append(example[5])
                    collated_dict["pos_docs_masks"].append(example[6])

            not_need_to_tensor_keys = ["sample_ids", "labels", "pos_docs_ids"]
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn


class RerankDataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, file_name, training=True, use_prefix=True, use_data_percent=1.0, is_sorted=True, n_candidates=16, use_gpt_as_gold=False) -> None:
        self.examples = []
        self.training = training
        self.is_sorted = is_sorted

        with open(file_name, encoding='utf-8') as f:
            data = f.readlines()

        n = len(data)
        n = int(use_data_percent * n)
        if n < len(data):
            random.seed(42)
            data = random.sample(data, n)

        for line in tqdm(data):
            input_ids = []
            record = json.loads(line)
            sample_id = record['sample_id']
            current_query = record['current_query']
            context = record['context']
            gold_rewrite = record['gold_rewrite']
            gold_answer = record['gold_answer']
            pos_docs_pids = record['pos_docs_pids']
            pos_docs_text = record['pos_docs_text']


            if use_prefix:
                current_query = "question:" + current_query
                first_context = True

            tokenized_query = tokenizer.encode(current_query, add_special_tokens = True, max_length = args.max_query_length)

            input_ids.extend(tokenized_query)
            for j in range(len(context) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length

                if use_prefix and first_context:
                    context[j] = "context: " + context[j]
                    first_context = False

                utt = tokenizer.encode(context[j], add_special_tokens=True, max_length= max_length, truncation=True)
                if len(input_ids) + len(utt) > args.max_concat_length:
                    input_ids += utt[:args.max_concat_length - len(input_ids) - 1] + [utt[-1]] # must ended with [SEP]
                    break
                else:
                    input_ids.extend(utt)

            input_ids, input_ids_mask = padding_seq_to_same_length(input_ids, max_pad_length=args.max_concat_length)

            if self.training:
                candidates = record['candidates']
                if self.is_sorted:
                    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                candidates = candidates[:n_candidates]

                if args.output_type == "rewrite":
                    if use_gpt_as_gold:
                        target_seq = [record["chatgpt_rewrite"]]
                    else:
                        target_seq = [gold_rewrite]
                elif args.output_type == "answer":
                    target_seq = [gold_answer]

                for candidate in candidates:
                    target_seq.append(candidate[0])

                # target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)
                # labels = target_encoding.input_ids
                # labels = torch.tensor(labels)
                # labels[labels == tokenizer.pad_token_id] = -100
                # labels = labels.tolist()

                if args.output_type == "rewrite":
                    decoder_inputs = tokenizer.batch_encode_plus(target_seq, max_length=args.max_query_length, padding="max_length", truncation=True)["input_ids"]
                elif args.output_type == "answer":
                    decoder_inputs = tokenizer.batch_encode_plus(target_seq, max_length=args.max_response_length, padding="max_length", truncation=True)["input_ids"]

                # Because T5 tokenizer would not add start_token defaultly, we should add mannually.
                decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
                _decoder_inputs = decoder_inputs.new_zeros(decoder_inputs.size(0), decoder_inputs.size(1) + 1)
                _decoder_inputs[:, 1:] = decoder_inputs.clone()
                _decoder_inputs[:, 0] = tokenizer.pad_token_id
                decoder_inputs = _decoder_inputs.tolist()

                for idx in range(len(pos_docs_pids)):
                    pos_docs = []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)

                    self.examples.append([sample_id,
                                        input_ids,
                                        input_ids_mask,
                                        decoder_inputs,
                                        target_seq,
                                        pos_docs,
                                        pos_docs_mask
                                        ])
            else:
                if args.output_type == "rewrite":
                    if use_gpt_as_gold:
                        target_seq = record["chatgpt_rewrite"]
                    else:
                        target_seq = gold_rewrite
                else:
                    target_seq = gold_answer

                if len(pos_docs_pids) == 0:
                    continue

                self.examples.append([sample_id,
                                    input_ids,
                                    input_ids_mask,
                                    target_seq,
                                    pos_docs_pids
                                    ])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch: list):
            collated_dict = {
                "sample_ids": [],
                "input_ids": [],
                "input_masks": [],
                "candidate_ids": [],
                "pos_docs_ids": [],
                "pos_docs_masks": [],
                "candidates": [],
            }

            for example in batch:
                if len(example) == 5:
                    collated_dict["sample_ids"].append(example[0])
                    collated_dict["input_ids"].append(example[1])
                    collated_dict["input_masks"].append(example[2])
                    collated_dict["candidates"].append(example[3])
                    collated_dict["pos_docs_ids"].append(example[4])
                else:
                    collated_dict["sample_ids"].append(example[0])
                    collated_dict["input_ids"].append(example[1])
                    collated_dict["input_masks"].append(example[2])
                    collated_dict["candidate_ids"].append(example[3])
                    collated_dict["candidates"].append(example[4])
                    collated_dict["pos_docs_ids"].append(example[5])
                    collated_dict["pos_docs_masks"].append(example[6])

            not_need_to_tensor_keys = ["sample_ids", "candidates", "pos_docs_ids"]
            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)

            return collated_dict

        return collate_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_type", type=str, default="rewrite")
    parser.add_argument("--num_train_epochs", type=int, default=15, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

    file_name = "/home/init0xyz/InfoCQR/datasets/qrecc/toys_candidates.json"

    test_dataset = RerankDataset_qrecc(args, tokenizer, file_name, training=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=test_dataset.get_collate_fn(args))

    for batch in tqdm(test_loader):
        print(batch)
        break
