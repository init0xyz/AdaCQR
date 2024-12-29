import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from shared_utils.data import RewriterDataset_qrecc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_checkpoint_path", type=str, default="/data/laiyilong/InfoCQR/checkpoints/t5_rerank_rewrite_with_gpt_margin_0dot1_retrain_with_sprse_dense/epoch=0-step=3798.ckpt")
    parser.add_argument("--input_file_path", type=str, default="/data/laiyilong/InfoCQR/datasets/qrecc/train_candidates_with_sparse_dense_score.json")
    parser.add_argument("--output_file_path", type=str, default="/data/laiyilong/InfoCQR/output/qrecc_train_candidates_stage3.json")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_type", type=str, default="rewrite")
    parser.add_argument("--model_type", type=str, default="T5")
    parser.add_argument("--use_last_response", type=bool, default=False)
    parser.add_argument("--use_prefix", type=bool, default=True)

    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--per_gpu_eval_batch_size", type=int,  default=4)
    parser.add_argument("--use_data_percent", type=float, default=1)

    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="qrecc")
    parser.add_argument("--n_gpu", type=int, default=1)

    parser.add_argument("--cand_num", type=int, default=32)

    args = parser.parse_args()

    device = torch.device(f"cuda:{str(args.gpu_id)}" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args

def main(args):
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

    old_state_dict = torch.load(args.model_checkpoint_path, map_location=args.device)['state_dict']
    # remove_prefix = 'model.'
    remove_prefix = 'model.model.'
    old_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in old_state_dict.items()}
    model.load_state_dict(old_state_dict)
    model.to(args.device)

    if args.task == "qrecc":
        test_dataset = RewriterDataset_qrecc(args, tokenizer, args.input_file_path, training=False, use_gpt_as_gold=True)
    else:
        raise NotImplementedError()

    args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_dataloader = DataLoader(test_dataset,
                                  shuffle=False,
                                  batch_size=args.batch_size,
                                  collate_fn=test_dataset.get_collate_fn(args))

    with open(args.output_file_path, "w") as f:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader, desc="Step"):
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["input_masks"].to(args.device)
                output_seqs = model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            num_return_sequences=args.cand_num,
                                            num_beam_groups=args.cand_num,
                                            diversity_penalty=2.0,
                                            num_beams=args.cand_num,
                                            # num_beams=16,
                                            length_penalty=1.0,
                                            early_stopping=True,
                                            no_repeat_ngram_size=3,
                                            min_length=8,
                                            max_length=args.max_query_length,
                                            )

                outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)

                for i in range(0, len(outputs), args.cand_num):
                    record = {}
                    idx = i // args.cand_num
                    record["sample_id"] = batch["sample_ids"][idx]
                    record["candidates"] = outputs[i:i+args.cand_num]
                    record["gold_labels"] = batch["labels"][idx]

                    f.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    args = get_args()
    main(args)


