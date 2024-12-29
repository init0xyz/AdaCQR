from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from modeling_t5 import T5Scorer


class RankingLoss(torch.nn.Module):
    def __init__(self, margin=0.001, gold_margin=0, gold_weight=0):
        super().__init__()
        self.margin = margin
        self.gold_margin = gold_margin
        self.gold_weight = gold_weight

    def forward(self, scores, summary_score):
        ones = torch.ones_like(scores)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        totalloss = loss_func(scores, scores, ones)
        # candidate loss
        for i in range(1, scores.size(1)):
            positive_score = scores[:, :-i]
            negative_score = scores[:, i:]
            positive_score = positive_score.contiguous().view(-1)
            negative_score = negative_score.contiguous().view(-1)
            ones = torch.ones_like(positive_score)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            loss = loss_func(positive_score, negative_score, ones)
            totalloss += loss

        # predicted summary loss
        positive_score = summary_score.unsqueeze(-1).expand_as(scores)
        negative_score = scores
        positive_score = positive_score.contiguous().view(-1)
        negative_score = negative_score.contiguous().view(-1)
        ones = torch.ones_like(positive_score)
        loss_func = torch.nn.MarginRankingLoss(self.gold_margin)
        totalloss += self.gold_weight * loss_func(positive_score, negative_score, ones)
        return totalloss

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super().__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, inputs, target):
        inputs = inputs.transpose(1, 2)  # [batch_size, seq_len, word_num]
        inputs = torch.log_softmax(inputs, dim=2)

        k = inputs.size(2)
        target_prob = torch.ones_like(inputs).type_as(inputs) * self.epsilon * 1 / k

        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))

        loss = - torch.mul(target_prob, inputs)
        loss = loss.sum(2)

        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(inputs)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()

        return loss

class RerankModel(nn.Module):
    def __init__(self, pad_token_id, model_name="google-t5/t5-base", cache_dir="./cache") -> None:
        super().__init__()
        self.model = T5Scorer.from_pretrained(model_name, cache_dir=cache_dir)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, input_masks, candidate_ids, normalize=True, score_mode="base", length_penalty=1.0, require_gold=True, adding=0):
        batch_size = input_ids.size(0)

        cand_masks = candidate_ids != self.pad_token_id
        cand_masks[:, :, 0] = 1
        output = self.model(
            input_ids=input_ids,
            attention_mask=input_masks,
            decoder_input_ids=candidate_ids,
            decoder_attention_mask=cand_masks,
            output_hidden_states=True
        )

        output = output[0]  # [bz x cand_num, seq_len, word_dim]
        output = output.view(batch_size, -1, output.size(1), output.size(2)) # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]
        output = output[:, :, :-1]  # truncate last token
        candidate_ids = candidate_ids[:, :, 1:]  # shift right
        cand_masks = candidate_ids != self.pad_token_id
        candidate_ids = candidate_ids.unsqueeze(-1)
        if normalize:
            if score_mode == "log":
                _output = F.log_softmax(output, dim=3)
            else:
                _output = F.softmax(output, dim=3)
            scores = torch.gather(_output, 3, candidate_ids).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_ids).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_masks.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1) + adding) ** length_penalty) # [bz, cand_num]
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}
        else:
            output = {'score': scores, "probs": probs}
        return output

    def scoring_mode(self):
        self.model.scoring_mode()

    def generation_mode(self):
        self.model.generation_mode()

    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
        do_sample: bool | None = None,
        early_stopping: bool | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        bad_words_ids: Iterable[int] | None = None,
        bos_token_id: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        length_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        encoder_no_repeat_ngram_size: int | None = None,
        num_return_sequences: int | None = None,
        max_time: float | None = None,
        decoder_start_token_id: int | None = None,
        use_cache: bool | None = None,
        num_beam_groups: int | None = None,
        diversity_penalty: float | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_scores: bool | None = None,
        return_dict_in_generate: bool | None = None,
        forced_bos_token_id: int | None = None,
        forced_eos_token_id: int | None = None,
        remove_invalid_values: bool | None = None,
        synced_gpus: bool | None = None,
        **model_kwargs,
    ):
        return self.model.generate(input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            max_time=max_time,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            remove_invalid_values=remove_invalid_values,
            synced_gpus=synced_gpus,
            **model_kwargs)

