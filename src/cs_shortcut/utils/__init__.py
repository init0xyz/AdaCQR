import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from pytz import timezone


def batch_to_device(batch, device):
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue

        batch[k] = v.to(device)
    return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

def get_logger(name=None):
    if not name:
        name = 'main'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    def customTime(*args):
        utc_dt = datetime.datetime.now()
        my_tz = timezone("Europe/London")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
    logging.Formatter.converter = customTime
    return logger


def get_has_qrel_label_sample_ids(qrel_file):
    # with open(qrel_file, 'r') as f:
    #     qrel_data = f.readlines()
    # qids = set()
    # for line in qrel_data:
    #     line = line.strip().split("\t")
    #     if len(line) == 1:
    #         line = line[0].strip().split(' ')
    #     qid = line[0]
    #     qids.add(qid)
    qrels = json.load(open(qrel_file))
    return list(qrels.keys())


def get_finished_sample_ids(output_file_path):
    finished_samples = {}
    if os.path.exists(output_file_path):
        with open(output_file_path) as f:
            data = f.readlines()
        for line in data:
            line = json.loads(line)
            finished_samples[line['sample_id']] = {}
            if "predicted_rewrite" in line:
                finished_samples[line['sample_id']]["predicted_rewrite"] = line['predicted_rewrite']
            if "predicted_response" in line:
                finished_samples[line['sample_id']]["predicted_response"] = line['predicted_response']
            if "cot" in line:
                finished_samples[line['sample_id']]["cot"] = line['cot']
            if "rewrite_part_text" in line:
                finished_samples[line['sample_id']]["rewrite_part_text"] = line['rewrite_part_text']

    return finished_samples
