
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict

from sklearn.model_selection import train_test_split

from src.utils import getDF
from src.path import RAW_DATA_DIR

class NGCFDataLoader:
    def __init__(
        self,
        fname,
        source: str = "amazon",
        test_size: float = 0.2,
        seed: int = 42

    ):
        self.fname = fname
        self.fpath = RAW_DATA_DIR / f"{fname}.jsonl.gz"
        self.source = source
        self.test_size = test_size
        self.seed = seed

        self.raw_df = self._load_data(self.fpath)
        (
            self.user2id,
            self.item2id,
            self.user_num,
            self.item_num,
            self.train_df,
            self.test_df
        ) = self._get_interaction_data()
        self.R = self._build_R()
        self.L = self._build_L()
        self.user_pos = self._get_user_pos()



    def _load_data(self, path):
        return getDF(path).rename(
            columns = {
                "user_id": "user",
                "asin": "item"
            }
        )[["user", "item"]]

    def _get_interaction_data(self):
        df = self.raw_df.copy()
        df = df.drop_duplicates(subset = ["user", "item"])

        user2id = {u: idx for idx, u in enumerate(df["user"].unique())}
        item2id = {i: idx for idx, i in enumerate(df["item"].unique())}

        df["user_idx"] = df["user"].map(user2id)
        df["item_idx"] = df["item"].map(item2id)

        user_num, item_num = df["user_idx"].max() + 1, df["item_idx"].max() + 1

        train_df, test_df = train_test_split(
            df[["user_idx", "item_idx"]],
            test_size = self.test_size,
            random_state = self.seed
        )
        return user2id, item2id, user_num, item_num, train_df, test_df

    def _build_R(self): # Adjacency Matrix
        R = sp.dok_matrix((self.user_num, self.item_num), dtype = np.float32)

        for _, (u, i) in self.train_df.iterrows():
            R[u, i] = 1.0
        return R

    def _build_L(self): # Laplacian Matrix
        R = self.R.tocsr() # for fast operations

        # sparse zero blocks
        zero_uu = sp.csr_matrix((self.user_num, self.user_num), dtype=np.float32)
        zero_ii = sp.csr_matrix((self.item_num, self.item_num), dtype=np.float32)
        print(zero_uu.shape)
        print(self.R.shape)

        # top = np.concat([zero_uu, self.R], axis = 1)
        # bottom = np.concat([self.R.T, zero_ii], axis = 1)
        # A = np.concat([top, bottom], axis = 0)
        top = sp.hstack([zero_uu, R], format="csr")
        bottom = sp.hstack([R.T, zero_ii], format="csr")
        A = sp.vstack([top, bottom], format="csr") 

        # degree
        d = np.array(A.sum(axis=1)).flatten()          # (N,)
        D_inv_sqrt = sp.diags(np.power(d + 1e-8, -0.5) )                  # sparse diagonal

        L = D_inv_sqrt @ A @ D_inv_sqrt # Normalized Laplacian Matrix
        L = L.tocoo().astype(np.float32)

        indices = torch.from_numpy(
            np.vstack((L.row, L.col)).astype(np.int64)
        )
        values = torch.from_numpy(L.data)
        shape = torch.Size(L.shape)

        return torch.sparse.FloatTensor(indices, values, shape) # Sparse Tensor
    
    def _get_user_pos(self):
        user_pos = defaultdict(set)

        for _, (u, i) in self.train_df.iterrows():
            user_pos[u].add(i)
        return user_pos
    
    def get_bpr_batch(self, batch_size: int):
        """
        sample a mini-batch of (user, pos_item, neg_item) triplets for BPR training

        returns:
            users: Tensor of shape (batch_size,)
            pos_items: Tensor of shape (batch_size,)
            neg_items: Tensor of shape (batch_size,)
        """
        users = []
        pos_items = []
        neg_items = []

        all_times = np.arange(self.item_num)

        for _ in range(batch_size):
            # 1) randomly sample a user at least on interaction
            u = np.random.choice(list(self.user_pos.keys()))
            i = np.random.choice(list(self.user_pos[u]))

            # 2) randomly sample a negative item
            while True:
                j = np.random.choice(all_times)
                if j not in self.user_pos[u]:
                    break
            
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)

        return (
            torch.LongTensor(users), 
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items)
        )