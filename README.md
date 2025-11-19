# **NGCF-Pytorch**

PyTorch implementation of **NGCF (Neural Graph Collaborative Filtering)**  
for studying graph-based collaborative filtering on user–item interaction data:

> **Neural Graph Collaborative Filtering**  
> *Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, Tat-Seng Chua (SIGIR 2019)*  

---

## 📂 **Project Structure**

```bash
NGCF_pytorch/
│
├── model/
│   └── ngcf.py          # NGCF model, evaluator, trainer
│
├── src/
│   ├── data.py          # NGCFDataLoader, graph builder, BPR sampler
│   ├── path.py          # project path manager (BASE_DIR, DATA_DIR, etc.)
│   ├── utils.py         # JSONL loader, YAML loader, seed utilities
│   └── config.yaml      # all hyperparameters & experiment settings
│
├── data/
│   └── raw/
│       └── <fname>.jsonl.gz   # Amazon-style input (user_id, asin)
│
├── main.py              # main training & test script
└── requirements.txt
````

---

All hyperparameters are centralized in `src/config.yaml`:

```yaml
data:
  fname: "Subscription_Boxes"  # RAW_DATA_DIR/<fname>.jsonl.gz
  source: "amazon"
  test_size: 0.2
  seed: 42

model:
  embed_dim: 64
  n_layer: 2
  dropout: 0.1
  l2_reg: 0.0001
  negative_slope: 0.2

train:
  batch_size: 1024
  epoch_num: 400
  num_batches_per_epoch: 200
  lr: 0.001
  eval_interval: 5
  patience: 10

eval:
  k: 10
  num_neg: 100
  user_sample_size: 10000

path:
  best_model_path: "best_ngcf_model.pth"
```

---

## **Run Training & Test**

1. Place the gzipped JSONL file under:

```bash
data/raw/Subscription_Boxes.jsonl.gz
# (or another name matching data.fname in config.yaml)
```

The file should contain at least:

* `user_id`
* `asin`

2. Install dependencies:

```bash
uv pip install -r requirements.txt
```

3. Run main script:

```bash
uv run main.py
```

During training you’ll see logs like:

```text
[INFO] Using device: cuda
[INFO] Loading data...
[INFO] #users: 123, #items: 456
[INFO] #train interactions: ...
[INFO] #val   interactions: ...
[INFO] #test  interactions: ...

[Epoch 10] Train Loss: 0.4821
Eval - NDCG@10: 0.3210, Hit@10: 0.6120
  ** Best model updated and saved to 'best_ngcf_model.pth' **

[INFO] Loading best model and evaluating on TEST set...
========================================
[TEST] NDCG@10: 0.3456, Hit@10: 0.6300
```

---

## **Evaluation Metrics**

The repository implements the common top-K metrics for recommendation:

* **NDCG@K**
* **Hit Ratio@K**

Evaluation follows the NGCF setting:

* Leave-one-out evaluation
* For each user:

  * 1 positive target item (from val/test)
  * 100 negative samples
  * Rank among 1 + 100 candidates and compute NDCG@K, Hit@K.

---

## **Acknowledgements**

Inspired by the original NGCF paper and official implementations in the recommender systems community.

```
* [https://github.com/xiangwang1223/neural_graph_collaborative_filtering](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
```
