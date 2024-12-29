# AdaCQR: Enhancing Query Reformulation for Conversational Search via Sparse and Dense Retrieval Alignment(COLING 2025)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT) [![Arxiv](https://img.shields.io/badge/arXiv-2407.01965-B21A1B)](https://arxiv.org/abs/2407.01965)

## 💻 Codes
### Passages Indexing
See [README.md](https://github.com/init0xyz/AdaCQR-preview/blob/main/preprocess/README.md) in preprocess folder.

### Training
NOTE: The processed training and testing datasets, including QReCC and TopiOCQA, can be downloaded from [Google Drive](https://drive.google.com/file/d/1Rp6xSUbJ0apAB9Bhikl3wVsPUluwb-R0/view?usp=sharing).

Before training and inference, please modify the provided scripts to make sure the variables(including index path and data path) are correctly set.

- **Stage 1 Training:**
```bash
bash scripts/train_stage1.sh
```

After obtaining Stage 1 models, modify the checkpoint path and output path in ```scripts/gen_candidates.sh```, then **generating candidates for Stage 2:**
```bash
bash scripts/gen_candidates.sh
```

Leveraging the generated candidates, we use a fusion metric to **obtain the relative orders for Stage 2 training**:
```bash
bash scripts/obtain_ranking_ance.sh
bash scripts/obtain_ranking_bm25.sh
bash scripts/ranking_fusion.sh
```

- **Stage 2 Training:**
```bash
bash scripts/train_rerank.sh
```
We manually stop Stage 2 training after one epoch and leverage the new checkpoints to generate new candidates for training, which is needed 2-3 times based on our experiments.

### Inference
- For inference of AdaCQR, the script is being used to generate reformulation queries:
```bash
bash scripts/test_rerank.sh
```
- For BM25 and ANCE retrieval, we provide ```src/cs_shortcut/run_dense_search.sh``` and ```src/test_BM25_direct.py```, feel free to use them.

- For Query Expansion, you can use tools like [vLLM](https://docs.vllm.ai/en/latest/) or [Ollama](https://ollama.com/) to leverage LLM to generate pseudo expansion.

## 📖 Citation
```bibtex
@misc{lai2024adacqrenhancingqueryreformulation,
      title={AdaCQR: Enhancing Query Reformulation for Conversational Search via Sparse and Dense Retrieval Alignment}, 
      author={Yilong Lai and Jialong Wu and Congzhi Zhang and Haowen Sun and Deyu Zhou},
      year={2024},
      eprint={2407.01965},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01965}, 
}
```

## Acknowledgment
We are very grateful to leverage prior works & source code to build this work, which includes [ConvGQR](https://github.com/fengranMark/ConvGQR), [InfoCQR](https://github.com/smartyfh/InfoCQR), [cs-shortcut](https://github.com/naver-ai/cs-shortcut), [LLM4CS](https://github.com/kyriemao/LLM4CS), [BRIO](https://github.com/yixinL7/BRIO).

If you have any question about AdaCQR, feel free to contact me by [yilong.lai@seu.edu.cn](mailto:yilong.lai@seu.edu.cn).
