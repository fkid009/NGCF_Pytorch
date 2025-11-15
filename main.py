# main_ngcf.py
import os

import torch
import torch.optim as optim

from src.path import ROOT_DIR
from src.utils import load_yaml, set_seed
from src.data import NGCFDataLoader  
from model.ngcf import NGCF, trainer_ngcf


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Load Config
    # -------------------------------------------------------------------------
    CONFIG_FPATH = ROOT_DIR / "config_ngcf.yaml"  # or reuse "config.yaml" if you want
    cfg = load_yaml(CONFIG_FPATH)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    es_cfg = cfg["early_stopping"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]

    # -------------------------------------------------------------------------
    # Device & Seed
    # -------------------------------------------------------------------------
    req_dev = train_cfg.get("device", "cpu")
    if req_dev == "cuda" and not torch.cuda.is_available():
        print("[INFO] cuda is not available, switched to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(req_dev)
    train_cfg["device"] = str(device)

    set_seed(train_cfg.get("seed", 42))

    # Ensure checkpoint directory exists
    best_model_path = train_cfg["best_model_path"]
    best_model_dir = os.path.dirname(best_model_path)
    if best_model_dir:
        os.makedirs(best_model_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Preparation (NGCFDataLoader)
    # -------------------------------------------------------------------------
    data_loader = NGCFDataLoader(
        fname=data_cfg["fname"],
        source=data_cfg.get("source", "amazon"),
        test_size=data_cfg.get("test_size", 0.2),
        val_size=data_cfg.get("val_size", 0.1),
        random_state=data_cfg.get("random_state", 42),
    )

    user_num = data_loader.user_num
    item_num = data_loader.item_num

    print(f"[INFO] #users={user_num}, #items={item_num}")
    print(f"[INFO] train={len(data_loader.train)}, val={len(data_loader.val)}, test={len(data_loader.test)}")

    # -------------------------------------------------------------------------
    # Model Initialization (NGCF)
    # -------------------------------------------------------------------------
    model = NGCF(
        user_num=user_num,
        item_num=item_num,
        norm_adj=data_loader.norm_adj,  # scipy sparse matrix
        embed_dim=model_cfg.get("embed_dim", 64),
        layer_sizes=model_cfg.get("layer_sizes", [64]),
        regs=tuple(model_cfg.get("regs", [1e-5])),
        node_dropout=model_cfg.get("node_dropout", 0.0),
        mess_dropout=model_cfg.get("mess_dropout", 0.1),
        device=str(device),
    ).to(device)

    # Make sure trainer can access model.device
    model.device = device

    # -------------------------------------------------------------------------
    # Optimizer (BPR loss is inside the model, so no criterion needed)
    # -------------------------------------------------------------------------
    opt_name = opt_cfg["name"].lower()
    if opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    else:
        raise NotImplementedError(f"Optimizer '{opt_cfg['name']}' is not implemented.")

    # -------------------------------------------------------------------------
    # Training (with Early Stopping & Best Model Saving)
    # -------------------------------------------------------------------------
    trainer_ngcf(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        num_epochs=train_cfg["num_epochs"],
        num_batches=train_cfg["num_batches_per_epoch"],
        batch_size=train_cfg["batch_size"],
        eval_interval=train_cfg.get("eval_interval", 2),
        patience=es_cfg["patience"],
        min_delta=es_cfg.get("min_delta", 0.0),
        best_model_path=train_cfg["best_model_path"],
        K=train_cfg.get("topk", 10),
        num_users_eval=train_cfg.get("num_users_eval", 10000),
        num_neg=train_cfg.get("num_neg_eval", 100),
    )

    print("[DONE] NGCF training finished.")
