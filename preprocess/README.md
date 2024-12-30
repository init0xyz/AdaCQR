# Passages Indexing
The passages indexing of TopiOCQA, QReCC, TREC CAsT 2019 & 2020, and TREC CAsT 2021 are time-consuming and take up a lot of disk space. 

Nevertheless, we provide codes and scripts in ```src/cs_shortcut``` to build the pipeline for passages indexing. In detail, the ```data_preprocessing.sh``` will scan all available jsonline passages:
```python
if args.task == "qrecc":
    passage_files = glob(f"{args.test_collection_path}/collection-paragraph/*/*.jsonl")
elif args.task == "topiocqa":
    passage_files = glob(f"{args.test_collection_path}/collections/*.jsonl")
elif args.task == "cast":
    passage_files = glob(f"{args.test_collection_path}/collections/*.jsonl")
elif args.task == "cast21":
    passage_files = glob(f"{args.test_collection_path}/collections/*.jsonl")
```
So modify the variables and run the ```data_preprocessing.sh``` to process all the passages:
```bash
bash src/cs_shortcut/data_preprocessing.sh
```
Then, you will obtain ```data.h5``` and some other files for each dataset. The dense passages index is built using the following command:
```bash
bash src/cs_shortcut/build_dense_index.sh
```
For BM25 indexing, use ```src/cs_shortcut/topiocqa_convert_to_pyserini_file.py``` to convert files to pyserini files. See [Pyserini Doc](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-embeddable-python-implementation) for more details.

Some useful links for dataset downloading and preprocessing:
- TopiOCQA: [link1](https://github.com/McGill-NLP/topiocqa/), [link2](https://github.com/naver-ai/cs-shortcut/)
- QReCC: [link1](https://github.com/apple/ml-qrecc), [link2](https://github.com/naver-ai/cs-shortcut/)
- TREC CAsT 2019 & 2020: [link1](https://www.treccast.ai/), [link2](https://github.com/kyriemao/ConvTrans/), [link3](https://github.com/thunlp/ConvDR)
- TREC CAsT 2021: [link1](https://www.treccast.ai/), [link2](https://github.com/grill-lab/trec-cast-tools/)