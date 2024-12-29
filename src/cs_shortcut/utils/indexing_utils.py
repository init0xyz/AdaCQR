import json
import logging
import os
import pickle
import random

import faiss
import h5py
import numpy as np
from numpy import linalg as LA

logger = logging.getLogger(__name__)

def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)


class DocumentCollection:
    def __init__(self,
                 data_path,
                 pid_file_path=None,
                 max_passage_length=384,
                 seed=42):
        self.data_path = data_path
        self.max_passage_length = max_passage_length
        self.file = None
        self.length = 0

        if pid_file_path is not None:
            self.pid_file_path = pid_file_path
        else:
            self.pid_file_path = None

        if os.path.exists(data_path):
            self.data_path = data_path
            self.file = h5py.File(data_path, "r")
            self.length = len(self.file["data"])

        self.rng = random.Random(seed)

    def write_h5(self,
                 passage_files,
                 ctx_tokenizer,
                 logging_step=100000):

        if self.pid_file_path is None:
            raise NotImplementedError("Writing h5 file must have pid_file_path!")

        file = h5py.File(f"{self.data_path}", "w")
        f = file.create_dataset("data",
                                dtype=h5py.string_dtype(),
                                shape=(100, ),
                                chunks=True,
                                maxshape=(None,),
                                compression="gzip")

        instances = []
        pid2index = {}
        h5_index = 0
        for passage_file in passage_files:
            for _, line in enumerate(open(passage_file, encoding="utf-8")):
                example = json.loads(line)
                # text = example["contents"]
                text = example["content"]
                example_id = example["id"]

                i = {"text": text, "id": example_id}
                instances.append(json.dumps(i))

                # pid2index[example_id] = h5_index
                pid2index[example_id] = example_id

                h5_index += 1

                if h5_index % 1000 == 0:
                    f.resize(h5_index, axis=0)
                    f[h5_index-1000:h5_index] = instances
                    instances = []

                if h5_index % logging_step == 0:
                    logger.info(f"Write passage data.h5 ... [{h5_index}]")

        if len(instances) > 0:
            f.resize(h5_index, axis=0)
            f[h5_index - len(instances):h5_index] = instances
            instances = []
        file.close()
        logger.info("Write passage data.h5 done!")

        pstore(pid2index, self.pid_file_path)
        logger.info(f"Write pid file to {self.pid_file_path} done!")

    def __len__(self):
        return self.length

    def get_data(self, id_):
        return json.loads(self.file["data"][id_])["input_id"]

    def get_text(self, id_):
        return json.loads(self.file["data"][id_])["text"]

    def get_pid(self, id_):
        return json.loads(self.file["data"][id_])["id"]

    def negative_sampling(self, negative_ids, k):
        negative_ids = self.rng.sample(negative_ids, k)
        return [self.get_data(i) for i in negative_ids]


class FaissIndex:

    def __init__(self, dim, path=None, device=-1, ann_search=False):
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.device = device
        self.ann_search = ann_search
        if self.ann_search:
            index = faiss.IndexHNSWFlat(self.dim, 512, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 128
            index.hnsw.efConstruction = 256
            self.index = index
            self.buffer = []

    def __len__(self):
        return self.index.ntotal

    def construct_hnsw_index(self, data, buffer_size=50000):
        n = len(data)
        bs = buffer_size
        for i in range(0, n, bs):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + bs]]
            vectors = np.concatenate(vectors, axis=0)
            self.index.add(vectors)
            logger.info(f"data hnsw indexed {self.index.ntotal}")
        logger.info(f"Total data hnsw indexed {self.index.ntotal}")

    def add(self, array):
        if self.ann_search:
            self.buffer.append(array)
        else:
            self.index.add(array)

    def save(self, path):
        if self.device >= 0:
            self.index = faiss.index_gpu_to_cpu(self.index)
            self.device = -1

        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, dim, path):
        faiss_index = cls(dim)
        index = faiss.read_index(path)
        faiss_index.index = index
        return faiss_index

    def to_cuda(self, device):
        self.device = device
        if self.device < 0:
            return

        faiss_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(faiss_res, self.device, self.index)
        self.index = index

    def search(self, queries, topk):
        return self.index.search(queries, topk)


class DenseIndexer:
    def __init__(self,
                 dim=768,
                 batch_size=128,
                 max_buffer_size=592000,
                 logger=None):
        self.dim = dim
        self.batch_size = batch_size
        self.index = None
        self.max_buffer_size = max_buffer_size
        self.logger = logger
        faiss.omp_set_num_threads(16)

    def load_index(self, index_path):
        index = FaissIndex.load(self.dim, index_path)
        self.index = index

        self.index.to_cuda(0)

        if self.logger:
            print(f"Loading precomputed index..! {len(self.index)} number of indices", self.logger)

    def save_index(self, save_path):
        if self.index:
            self.index.save(save_path)

    def passage_inference(self, ctx_encoder, sentences, sent_ids, split,
                          output_path=None, embeddings=None, multi_gpus=False, ann_search=False):
#         device = ctx_encoder.device
        if output_path:
            tmp = "/".join(output_path.split("/")[:-1])
            json.dump(sent_ids, open(f"{tmp}/doc_ids_{split}.json", "w"), indent=2)

        index = FaissIndex(self.dim, ann_search=ann_search)

        if self.logger:
            print("Start inference!")

        if not multi_gpus:
            embeddings = ctx_encoder.encode(sentences,
                                            batch_size=self.batch_size,
                                            show_progress_bar=True)
            # these embeddings are normalized, so we can use cosine similarity by inner product

        for t in range(3):
            embed_norm = LA.norm(embeddings[t])
            print(f"Norm of embedding: {embed_norm}")

        index.add(embeddings)

        if output_path:
            if index.ann_search:
                index.buffer = np.vstack(index.buffer)
                index.construct_hnsw_index(index.buffer)

            index.save(output_path)

        if self.logger:
           print(f"Indexing {len(index)} done.")

    def retrieve(self,
                 inputs,
                 qids,
                 psg_ids,
                 top_k=100):
        if not self.index:
            return {}

        D, I = self.index.search(inputs, top_k)
        score_dict = {}
        for scores, pids, qid in zip(D, I, qids, strict=False):
            score_dict[qid] = {psg_ids[pid]: float(score) for score, pid in zip(scores, pids, strict=False)}

        return score_dict
