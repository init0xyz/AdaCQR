import json
from argparse import ArgumentParser

import lightning.pytorch as pl
import numpy as np
import pytrec_eval
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pyserini.search.lucene import LuceneSearcher
from shared_utils.data import RewriterDataset_qrecc
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup


class Model(pl.LightningModule):
    def __init__(self, args, cache_dir="./cache") -> None:
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name, cache_dir=cache_dir)
        self.searcher = LuceneSearcher(self.args.bm25_index_path)
        self.searcher.set_bm25(self.args.bm25_k1, self.args.bm25_b)

        if self.args.test:
            self.out_file = open(self.args.out_file_path, "w")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
            num_warmup_steps=self.args.warmup_ratio*self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches)
        lr_scheduler = {
            'scheduler': scheduler,
            "interval": "step",
            'name': 'linear-lr'
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        input_ids, input_masks, label_ids = batch["input_ids"], batch["input_masks"], batch["label_ids"]

        output = self.model(
            input_ids=input_ids,
            attention_mask=input_masks,
            labels=label_ids
        )

        loss = output.loss

        self.log("train/loss", loss.detach().float())
        return loss

    def validation_step(self, batch, batch_idx):
        sample_ids, input_ids, input_masks, pids  = batch["sample_ids"], batch["input_ids"], batch["input_masks"], batch["pos_docs_ids"]
        if self.args.output_type == "answer":
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=input_masks,
                do_sample=True,
                num_beams=3,
                no_repeat_ngram_size=3,
                max_length=self.args.max_response_length
            )
        else:
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=input_masks,
                do_sample=False,
                max_length=self.args.max_query_length
            )

        outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        qrels, qrels_ndcg = {}, {}

        for query_id, pid_list in zip(sample_ids, pids, strict=False):
            if query_id not in qrels:
                qrels[query_id] = {}
            if query_id not in qrels_ndcg:
                qrels_ndcg[query_id] = {}

            for passage_id in pid_list:
                passage_id = str(passage_id)
                qrels[query_id][passage_id] = 1
                qrels_ndcg[query_id][passage_id] = 1

        hits = self.searcher.batch_search(outputs, sample_ids, k=100, threads=16)

        runs = {}
        for current_id in sample_ids:
            runs[current_id] = {}
            for i, item in enumerate(hits[current_id]):
                runs[current_id][item.docid] = int(200-i-1)

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.10", "recall.100"})
        result_dict = evaluator.evaluate(runs)

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
        ndcg_result = evaluator.evaluate(runs)

        map_res = np.average([v['map'] for v in result_dict.values()])
        mrr_res = np.average([v['recip_rank'] for v in result_dict.values()])
        r100_res = np.average([v['recall_100'] for v in result_dict.values()])
        r10_res = np.average([v['recall_10'] for v in result_dict.values()])
        ndcg_res = np.average([v['ndcg_cut_3'] for v in ndcg_result.values()])


        valid_info = {
            "eval/MAP": map_res,
            "eval/MRR": mrr_res,
            "eval/R@10": r10_res,
            "eval/R@100": r100_res,
            "eval/NDCG": ndcg_res,
            "eval/score": map_res + mrr_res + r100_res + r10_res + ndcg_res,
        }

        self.log_dict(valid_info, sync_dist=True)

    def test_step(self, batch, batch_idx):
        sample_ids, input_ids, input_masks = batch["sample_ids"], batch["input_ids"], batch["input_masks"]
        if self.args.output_type == "answer":
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=input_masks,
                do_sample=True,
                num_beams=3,
                no_repeat_ngram_size=3,
                max_length=self.args.max_response_length
            )
        else:
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=input_masks,
                do_sample=False,
                num_beams=5,
                no_repeat_ngram_size=3,
                max_length=self.args.max_query_length
            )

        outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for cid, query in zip(sample_ids, outputs, strict=False):
            data = {}
            data["sample_id"] = cid
            if self.args.output_type == "answer":
                data["generated_answer"] = query
            else:
                data["generated_query"] = query
            self.out_file.write(json.dumps(data) + "\n")

def get_args():
    parser = ArgumentParser()

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--limit_val_batches", type=float, default=0.5)
    parser.add_argument("--val_check_interval", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--dev_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--out_file_path", type=str)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--gradient_accumulate", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--default_root_dir", type=str, default="./checkpoints/")

    parser.add_argument("--output_type", type=str, default="rewrite")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--bm25_index_path", type=str, default="/home/init0xyz/qrecc_bm25_index")
    parser.add_argument("--bm25_k1", type=float, default=0.82)
    parser.add_argument("--bm25_b", type=float, default=0.68)

    parser.add_argument("--use_gpt_as_gold", action="store_true", default=True)
    parser.add_argument("--model_name", type=str, default="google-t5/t5-base")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    args = get_args()

    if args.test is False:
        if args.wandb is True:
            logger = WandbLogger()
        else:
            logger = None
        lr_monitor = LearningRateMonitor(logging_interval="step")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        train_dataset = RewriterDataset_qrecc(args, tokenizer, args.train_data_path, training=True, use_gpt_as_gold=args.use_gpt_as_gold)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_dataset.get_collate_fn(args))

        dev_dataset = RewriterDataset_qrecc(args, tokenizer, args.dev_data_path, training=False)
        dev_loader = DataLoader(dev_dataset, batch_size=args.valid_batch_size, shuffle=True, collate_fn=dev_dataset.get_collate_fn(args))

        model = Model(args, args.checkpoint_path)

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.default_root_dir,
            save_top_k=8,
            save_weights_only=True,
            monitor="eval/score",
            save_on_train_epoch_end=False,
            enable_version_counter=False,
            verbose=True,
            mode="max"
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            limit_val_batches=args.limit_val_batches,
            log_every_n_steps=10,
            devices=args.devices,
            # precision=16,
            val_check_interval=args.val_check_interval,
            default_root_dir=args.default_root_dir,
            callbacks=[lr_monitor, checkpoint_callback],
            logger=logger,
            accumulate_grad_batches=args.gradient_accumulate,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        test_dataset = RewriterDataset_qrecc(args, tokenizer, args.test_data_path, training=False)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=test_dataset.get_collate_fn(args))

        model = Model(args)
        model.load_state_dict(torch.load(args.checkpoint_path)['state_dict'])
        trainer = pl.Trainer(devices=args.devices, precision="16-true", enable_checkpointing=False)
        # trainer = pl.Trainer(devices=args.devices, enable_checkpointing=False)
        trainer.test(model, dataloaders=test_loader)
