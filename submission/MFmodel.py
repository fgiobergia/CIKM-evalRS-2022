import numpy as np
import pandas as pd
from reclist.abstractions import RecModel

import time
import random

from tqdm import tqdm

# from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import NMF

from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix

class MyModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
    
    def train(self, train_df: pd.DataFrame):
        self.user_map = { k: v for v, k in enumerate(list(set(train_df["user_id"]))) }
        self.rev_user_map = { k: v for v,k in self.user_map.items() }
        self.track_map = { k: v for v, k in enumerate(list(set(train_df["track_id"]))) }
        self.rev_track_map = { k: v for v,k in self.track_map.items() }

        # self.mat = csr_matrix()
        data = []
        row_ind = []
        col_ind = []

        for _, row in train_df.iterrows():
            data.append(1) # use num_plays instead?
            row_ind.append(self.user_map[row.user_id])
            col_ind.append(self.track_map[row.track_id])
        
        self.mat = csr_matrix((data, (row_ind, col_ind)), shape=(len(self.user_map), len(self.track_map)))
        print("Matrix generated. Shape:", self.mat.shape)

        decomp = NMF(n_components=2, init="random")

        decomp.





    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:

        data = np.hstack([ user_ids["user_id"].values.reshape(-1, 1), preds ])
        return pd.DataFrame(data, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')

