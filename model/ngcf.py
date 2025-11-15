import random
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def scipy_to_torch_sparse(sparse_mtx) -> torch.Tensor:
    """
    Convert a scipy.sparse matrix to a torch.sparse.FloatTensor.
    """
    sparse_mtx = sparse_mtx.tocoo().astype("float32")
    indices = torch.from_numpy(
        np.vstack([sparse_mtx.row, sparse_mtx.col]).astype("int64")
    )
    values = torch.from_numpy(sparse_mtx.data)
    shape = torch.Size(sparse_mtx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class NGCF(nn.Module):
    """
    PyTorch implementation of NGCF, aligned with:
    - Original NGCF paper (SIGIR 2019)
    - Official TensorFlow implementation

    This version assumes:
    - DataLoader provides `user_num`, `item_num`
    - `norm_adj` is a scipy sparse matrix (D^{-1}(A + I) or D^{-1}A).
    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        norm_adj,              # scipy sparse matrix
        embed_dim: int = 64,
        layer_sizes=(64,),
        regs=(1e-5,),
        node_dropout=0.0,
        mess_dropout=0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.n_nodes = user_num + item_num

        self.embed_dim = embed_dim
        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(self.layer_sizes)

        self.regs = regs
        self.decay = regs[0]  # same as TF code: use first reg for embedding L2

        self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Convert adjacency to torch sparse and register as buffer
        # ------------------------------------------------------------------
        torch_norm_adj = scipy_to_torch_sparse(norm_adj).to(self.device)
        # Register as buffer so it's moved with .to(device) and saved in state_dict
        self.register_buffer("norm_adj", torch_norm_adj)

        # ------------------------------------------------------------------
        # Embedding parameters
        # ------------------------------------------------------------------
        self.user_embedding = nn.Embedding(user_num, embed_dim)
        self.item_embedding = nn.Embedding(item_num, embed_dim)

        # Layer dimensions: [input_dim, h1, h2, ...]
        self.weight_size_list = [embed_dim] + list(layer_sizes)

        # Graph convolution weights (sum and bi-interaction)
        self.W_gc = nn.ParameterList()
        self.b_gc = nn.ParameterList()
        self.W_bi = nn.ParameterList()
        self.b_bi = nn.ParameterList()

        for k in range(self.n_layers):
            in_dim = self.weight_size_list[k]
            out_dim = self.weight_size_list[k + 1]

            self.W_gc.append(nn.Parameter(torch.empty(in_dim, out_dim)))
            self.b_gc.append(nn.Parameter(torch.empty(1, out_dim)))

            self.W_bi.append(nn.Parameter(torch.empty(in_dim, out_dim)))
            self.b_bi.append(nn.Parameter(torch.empty(1, out_dim)))

        # Dropout settings: same dropout for all layers for simplicity
        self.node_dropout = node_dropout
        # allow list or scalar
        if isinstance(mess_dropout, (list, tuple)):
            assert len(mess_dropout) == self.n_layers
            self.mess_dropout = list(mess_dropout)
        else:
            self.mess_dropout = [mess_dropout] * self.n_layers

        # Initialize parameters (xavier)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for embeddings and weights."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

        for W in self.W_gc:
            nn.init.xavier_uniform_(W)
        for b in self.b_gc:
            nn.init.zeros_(b)

        for W in self.W_bi:
            nn.init.xavier_uniform_(W)
        for b in self.b_bi:
            nn.init.zeros_(b)

    # ------------------------------------------------------------------
    # Graph embedding propagation (NGCF)
    # ------------------------------------------------------------------
    def _dropout_sparse(self, x: torch.Tensor, keep_prob: float) -> torch.Tensor:
        """
        Dropout for torch.sparse.FloatTensor.
        """
        if keep_prob >= 1.0 or not self.training:
            return x

        # x._values() shape: (#nonzero,)
        noise = torch.rand(x._values().size(), device=x.device)
        dropout_mask = (noise < keep_prob).to(x.dtype)
        # Keep only selected values
        new_values = x._values() * dropout_mask / keep_prob

        # Remove zeroed-out entries to keep sparsity clean
        nonzero_mask = dropout_mask.nonzero(as_tuple=True)[0]
        new_indices = x._indices()[:, nonzero_mask]
        new_values = new_values[nonzero_mask]

        return torch.sparse.FloatTensor(new_indices, new_values, x.shape)

    def _create_ngcf_embed(self):
        """
        Perform NGCF-style embedding propagation over the whole graph.

        Returns:
            user_embeddings: (n_users, sum(layer_dims))
            item_embeddings: (n_items, sum(layer_dims))
        """
        # Apply node dropout on adjacency if needed
        if self.node_dropout > 0.0 and self.training:
            adj = self._dropout_sparse(self.norm_adj, keep_prob=1.0 - self.node_dropout)
        else:
            adj = self.norm_adj

        # Initial ego embeddings: concat[user, item]
        user_emb = self.user_embedding.weight    # (n_users, d)
        item_emb = self.item_embedding.weight    # (n_items, d)
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)  # (n_nodes, d)

        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            # Message passing: A * E
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)  # (n_nodes, d_k)

            # Sum-based message
            sum_embeddings = torch.matmul(side_embeddings, self.W_gc[k]) + self.b_gc[k]
            sum_embeddings = F.leaky_relu(sum_embeddings)

            # Bi-interaction message
            bi_interaction = ego_embeddings * side_embeddings
            bi_embeddings = torch.matmul(bi_interaction, self.W_bi[k]) + self.b_bi[k]
            bi_embeddings = F.leaky_relu(bi_embeddings)

            # Update ego embeddings
            ego_embeddings = sum_embeddings + bi_embeddings

            # Message dropout
            if self.mess_dropout[k] > 0.0:
                ego_embeddings = F.dropout(
                    ego_embeddings,
                    p=self.mess_dropout[k],
                    training=self.training,
                )

            # L2 normalize embeddings
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        # Concatenate embeddings from all layers
        all_embeddings = torch.cat(all_embeddings, dim=1)  # (n_nodes, d*(L+1))

        # Split back into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.user_num, self.item_num], dim=0
        )

        return user_all_embeddings, item_all_embeddings

    # ------------------------------------------------------------------
    # Forward for BPR learning
    # ------------------------------------------------------------------
    def forward(self, users, pos_items, neg_items):
        """
        Args:
            users: LongTensor of shape (batch_size,)
            pos_items: LongTensor of shape (batch_size,)
            neg_items: LongTensor of shape (batch_size,)
        Returns:
            u_emb, pos_emb, neg_emb: embeddings after NGCF propagation
        """
        # Compute graph-aware embeddings for all users/items
        user_all_emb, item_all_emb = self._create_ngcf_embed()

        # Select batch embeddings
        u_emb = user_all_emb[users]              # (B, D*)
        pos_emb = item_all_emb[pos_items]        # (B, D*)
        neg_emb = item_all_emb[neg_items]        # (B, D*)

        return u_emb, pos_emb, neg_emb

    # ------------------------------------------------------------------
    # BPR loss
    # ------------------------------------------------------------------
    def bpr_loss(self, users, pos_items, neg_items):
        """
        Compute BPR loss and L2 regularization term.
        """
        u_emb, pos_emb, neg_emb = self.forward(users, pos_items, neg_items)

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)  # (B,)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)  # (B,)

        # BPR loss: -log(sigmoid(pos - neg))
        mf_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        # L2 regularization on embeddings (same as TF code)
        reg_loss = (
            u_emb.norm(2).pow(2)
            + pos_emb.norm(2).pow(2)
            + neg_emb.norm(2).pow(2)
        ) / users.size(0)

        reg_loss = self.decay * reg_loss

        return mf_loss, reg_loss
    
def build_user_item_dict(df) -> Dict[int, list]:
    """
    Convert interaction DataFrame (user, item) into {user: [items]} dict.
    """
    return df.groupby("user")["item"].apply(list).to_dict()


def evaluate_ngcf(
    model,
    dataset,
    K: int = 10,
    num_users_eval: int = 10000,
    num_neg: int = 100,
    split: str = "val",
) -> Tuple[float, float]:
    """
    Evaluation for NGCF model using NDCG@K and Hit@K.

    Args:
        model: NGCF PyTorch model.
        dataset: tuple of
            (train_user_items, val_user_items, test_user_items, user_num, item_num)
        K: cutoff for NDCG@K and Hit@K.
        num_users_eval: number of users to sample for evaluation
                        (if fewer users exist, evaluate on all).
        num_neg: number of negative items to sample for each user.
        split: either "val" or "test".

    Returns:
        (NDCG@K, Hit@K)
    """
    (
        train_user_items,
        val_user_items,
        test_user_items,
        user_num,
        item_num,
    ) = dataset

    if split == "val":
        target_dict = val_user_items
    elif split == "test":
        target_dict = test_user_items
    else:
        raise ValueError("split must be 'val' or 'test'.")

    # Only users that have positive items in this split
    all_users = list(target_dict.keys())
    if len(all_users) == 0:
        return 0.0, 0.0

    # Sample a subset of users if too many
    if len(all_users) > num_users_eval:
        users = random.sample(all_users, num_users_eval)
    else:
        users = all_users

    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        # Compute graph-based embeddings for all users/items once
        user_all_emb, item_all_emb = model._create_ngcf_embed()
        user_all_emb = user_all_emb.to(device)
        item_all_emb = item_all_emb.to(device)

    NDCG, HR, valid_users = 0.0, 0.0, 0

    for u in users:
        # Training items for this user
        train_items = set(train_user_items.get(u, []))
        # Positives in current split
        pos_items = target_dict.get(u, [])
        if len(pos_items) == 0:
            continue

        # Choose one target item (similar style to your SASRec evaluation)
        target = random.choice(pos_items)

        # Build candidate set: 1 positive + num_neg negatives
        candidate_items = {target}
        # Avoid recommending items in train + current split
        forbidden_items = train_items.union(pos_items)

        while len(candidate_items) < num_neg + 1:
            neg = np.random.randint(0, item_num)
            if neg not in forbidden_items:
                candidate_items.add(neg)

        item_idx = list(candidate_items)

        # Make sure target is at index 0 for convenient rank computation
        if item_idx[0] != target:
            t_pos = item_idx.index(target)
            item_idx[0], item_idx[t_pos] = item_idx[t_pos], item_idx[0]

        # ---- Score computation ----
        u_emb = user_all_emb[u].unsqueeze(0)  # (1, D)
        item_tensor = torch.tensor(item_idx, dtype=torch.long, device=device)
        i_emb = item_all_emb[item_tensor]      # (num_items, D)

        scores = torch.sum(u_emb * i_emb, dim=1)  # (num_items,)
        scores = scores.detach().cpu().numpy()

        # Rank of the target item (index 0 in item_idx)
        preds = -scores  # higher score -> smaller value after negation
        ranks = preds.argsort().argsort()
        rank = int(ranks[0])

        valid_users += 1
        if rank < K:
            HR += 1
            NDCG += 1.0 / np.log2(rank + 2)

    if valid_users == 0:
        return 0.0, 0.0

    return NDCG / valid_users, HR / valid_users




def trainer_ngcf(
    model,
    data_loader,
    optimizer,
    num_epochs: int,
    num_batches: int,
    batch_size: int,
    eval_interval: int = 2,
    patience: int = 5,
    min_delta: float = 0.0,
    best_model_path: str = "best_ngcf_model.pth",
    K: int = 10,
    num_users_eval: int = 10000,
    num_neg: int = 100,
):
    """
    Trainer for NGCF with BPR loss, validation & test evaluation.

    Args:
        model: NGCF model (must define .bpr_loss and .device).
        data_loader: NGCFDataLoader instance.
        optimizer: torch optimizer.
        num_epochs: maximum number of epochs.
        num_batches: number of batches per epoch.
        batch_size: batch size for BPR sampling.
        eval_interval: evaluate every `eval_interval` epochs.
        patience: early stopping patience w.r.t. validation NDCG@K.
        min_delta: minimum improvement in validation NDCG@K to be considered as better.
        best_model_path: where to save the best model weights.
        K: cutoff for NDCG@K and Hit@K.
        num_users_eval: maximum number of users to evaluate per split.
        num_neg: number of negative items for each eval user.
    """
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    # -----------------------------------------------------
    # Build user->items dicts for evaluation
    # -----------------------------------------------------
    train_user_items = build_user_item_dict(data_loader.train)
    val_user_items = build_user_item_dict(data_loader.val)
    test_user_items = build_user_item_dict(data_loader.test)

    dataset = (
        train_user_items,
        val_user_items,
        test_user_items,
        data_loader.user_num,
        data_loader.item_num,
    )

    # Best metrics tracking
    best_val_ndcg = float("-inf")
    best_val_hr = 0.0
    best_test_ndcg = 0.0
    best_test_hr = 0.0

    # Early stopping counter
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_mf_loss = 0.0
        total_reg_loss = 0.0

        for _ in range(num_batches):
            # ---------------------------
            # BPR batch sampling
            # ---------------------------
            users, pos_items, neg_items = data_loader.get_bpr_batch(batch_size)

            users = torch.from_numpy(users).long().to(device)
            pos_items = torch.from_numpy(pos_items).long().to(device)
            neg_items = torch.from_numpy(neg_items).long().to(device)

            mf_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
            loss = mf_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mf_loss += mf_loss.item()
            total_reg_loss += reg_loss.item()

        avg_mf = total_mf_loss / max(1, num_batches)
        avg_reg = total_reg_loss / max(1, num_batches)
        print(
            f"[Epoch {epoch}] MF Loss: {avg_mf:.4f} | Reg Loss: {avg_reg:.4f} | Total: {avg_mf + avg_reg:.4f}"
        )

        # -------------------------------
        # Evaluation & early stopping
        # -------------------------------
        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_ndcg, val_hr = evaluate_ngcf(
                    model,
                    dataset,
                    K=K,
                    num_users_eval=num_users_eval,
                    num_neg=num_neg,
                    split="val",
                )
                test_ndcg, test_hr = evaluate_ngcf(
                    model,
                    dataset,
                    K=K,
                    num_users_eval=num_users_eval,
                    num_neg=num_neg,
                    split="test",
                )

            print(f"  Val  - NDCG@{K}: {val_ndcg:.4f}, Hit@{K}: {val_hr:.4f}")
            print(f"  Test - NDCG@{K}: {test_ndcg:.4f}, Hit@{K}: {test_hr:.4f}")

            # Check improvement on validation NDCG
            if val_ndcg > best_val_ndcg + min_delta:
                best_val_ndcg = val_ndcg
                best_val_hr = val_hr
                best_test_ndcg = test_ndcg
                best_test_hr = test_hr

                # Save best model weights
                torch.save(model.state_dict(), best_model_path)
                print(f"  ** Best model updated and saved to '{best_model_path}' **")

                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                print(f"  No improvement. Patience: {epochs_without_improve}/{patience}")

                if epochs_without_improve >= patience:
                    print("  >>> Early stopping triggered.")
                    break

    print("========================================")
    print(f"Best Validation : NDCG@{K}={best_val_ndcg:.4f}, Hit@{K}={best_val_hr:.4f}")
    print(f"Best Test       : NDCG@{K}={best_test_ndcg:.4f}, Hit@{K}={best_test_hr:.4f}")
    print(f"Best model weights saved at: {best_model_path}")