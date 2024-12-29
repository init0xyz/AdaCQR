
from transformers import BatchEncoding
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTokenizedInput, TensorType, TextInput, TruncationStrategy


class ConvTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self,
                 text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
                 text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
                 add_special_tokens: bool = True,
                 padding: bool | str | PaddingStrategy = False,
                 truncation: bool | str | TruncationStrategy = False,
                 max_length: int | None = None,
                 stride: int = 0,
                 is_split_into_words: bool = False,
                 pad_to_multiple_of: int | None = None,
                 return_tensors: str | TensorType | None = None,
                 return_token_type_ids: bool | None = None,
                 return_attention_mask: bool | None = None,
                 return_overflowing_tokens: bool = False,
                 return_special_tokens_mask: bool = False,
                 return_offsets_mapping: bool = False,
                 return_length: bool = False,
                 verbose: bool = True,
                 retain_first_utter: bool = False,
                 turn_delim_token: str = None,
                 **kwargs
                ) -> BatchEncoding:

        input_ids = []
        attention_mask = []
        if isinstance(text, str):
            text = [text]

        if not text_pair:
            text_pair = [None] * len(text)
        for context, uttr in zip(text, text_pair, strict=False):
            input_id, attn_mask = self._conv_encode(context,
                                                    uttr,
                                                    add_special_tokens,
                                                    padding,
                                                    truncation,
                                                    max_length,
                                                    retain_first_utter,
                                                    turn_delim_token)
            input_ids.append(input_id)
            attention_mask.append(attn_mask)

        if padding == PaddingStrategy.LONGEST.value:
            input_ids = self.pad_ids(input_ids, self.tokenizer.pad_token_id)
            attention_mask = self.pad_ids(attention_mask, 0)

        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask},
                             tensor_type=return_tensors)

    def pad_ids(self, arrays, padding, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [
            array + [padding] * (max_length - len(array))
            for array in arrays
        ]
        return arrays

    def _conv_encode(self,
                     context: str | list[str],
                     text: str = None,
                     add_special_tokens: bool = True,
                     padding: bool | str | PaddingStrategy = False,
                     truncation: bool | str | TruncationStrategy = False,
                     max_length: int | None = None,
                     retain_first_utter: bool = False,
                     turn_delim_token: str = None,
                    **kwargs
                ) -> BatchEncoding:

        if isinstance(context, str):
            context = [context]
            retain_first_utter = False

        first_token = []
        if retain_first_utter and len(context) > 1:
            first_token = self.tokenizer.tokenize(context[0])
            context = context[1:]

        context_tokens = []
        for utter in context:
            if context_tokens and turn_delim_token:
                context_tokens.append(turn_delim_token)
            context_tokens.extend(self.tokenizer.tokenize(utter))

        this_token = []
        if text:
            this_token = self.tokenizer.tokenize(text)

        max_length = max_length if max_length else min(self.tokenizer.model_max_length, 512)
        max_available_length = max_length
        if add_special_tokens:
            max_available_length -= 2

        max_context_length = max_available_length
        if first_token:
            max_context_length -= len(first_token)
            if turn_delim_token:
                max_context_length -= 1

        if this_token:
            max_context_length -= len(this_token)
            if turn_delim_token:
                max_context_length -= 1

        if max_context_length < 0 and first_token:
            max_context_length += len(first_token)
            if turn_delim_token:
                max_context_length += 1
            first_token = []

        if max_context_length < 0 and this_token:
            gap = len(this_token) - max_available_length
            assert gap >= 0
            this_token = this_token[gap:]
            first_token = []
            context_tokens = []

        if len(context_tokens) > max_context_length:
            gap = len(context_tokens) - max_context_length
            context_tokens = context_tokens[gap:]

        input_tokens = []
        if first_token:
            input_tokens = first_token

        if context_tokens:
            turn_sep = [turn_delim_token] if input_tokens and turn_delim_token else []
            input_tokens = input_tokens + turn_sep + context_tokens

        if this_token:
            turn_sep = [turn_delim_token] if input_tokens and turn_delim_token else []
            input_tokens = input_tokens + turn_sep + this_token

        if add_special_tokens and self.tokenizer.cls_token and self.tokenizer.sep_token:
            input_tokens = [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)

        while len(input_ids) < max_length and \
        (padding == PaddingStrategy.MAX_LENGTH.value or padding is True):
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)

        return input_ids, attention_mask
