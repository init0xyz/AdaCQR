import dataclasses
import json
import re
from dataclasses import dataclass

from tqdm import tqdm

EOT_TOKEN = "[SEP]"


def extract_split_from_file_name(file_name):
    check = re.search(r"(train|dev|test)", file_name)
    if check:
        return check.group()
    return ""


def load_processed_data(data_path):
    examples = []
    idx = 0
    with open(data_path, encoding="utf-8") as f:
        for line in tqdm(f):
            example = RetrievalInstance.from_dict(json.loads(line.strip("\n")))
            examples.append(example)
            idx += 1
    return examples


@dataclass
class RetrievalInstance:
    guid: str
    context: list[str]
    input_id: list[int]
    input_mask: list[int]
    cand_ids: list[list[int]]
    cand_mask: list[list[int]]
    label: int
    hard_negative_idx: int
    label_id: list[int] | None = None
    rewrite_id: list[int] | None = None
    hard_negative_ids: list[int] | None = None
    hard_negative_scores: list[int] | None = None
    has_positive: bool | None = True
    cons_hard_negative_ids: list[int] | None = None
    switch_hard_negative_ids: list[int] | None = None
    question_type: str | None = None

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)
