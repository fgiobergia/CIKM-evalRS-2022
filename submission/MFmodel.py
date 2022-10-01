import numpy as np
import sys
sys.path.insert(0,"..")
from evaluation.EvalRSRunner import ChallengeDataset
import pandas as pd
from reclist.abstractions import RecModel

from sklearn.decomposition import TruncatedSVD

import time
import random

from tqdm import tqdm

# from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import NMF

from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix

class MyModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
        self.top_k = top_k
    
    def train(self, train_df: pd.DataFrame):
        self.user_map = { k: v for v, k in enumerate(list(set(train_df["user_id"]))) }
        self.rev_user_map = { k: v for v,k in self.user_map.items() }
        self.track_map = { k: v for v, k in enumerate(list(set(train_df["track_id"]))) }
        self.rev_track_map = { k: v for v,k in self.track_map.items() }

        t1 = time.time()
        data = []
        row_ind = []
        col_ind = []

        self.train_df = train_df

        for _, row in train_df.iterrows():
            data.append(np.log(1+row.user_track_count)) # use num_plays instead?
            row_ind.append(self.user_map[row.user_id])
            col_ind.append(self.track_map[row.track_id])
        
        self.mat = csr_matrix((data, (row_ind, col_ind)), shape=(len(self.user_map), len(self.track_map)))
        t2 = time.time()
        print("Matrix generated in ", round(t2-t1,2), "seconds. Shape:", self.mat.shape)


        svd = TruncatedSVD(128)

        Us = svd.fit_transform(self.mat)
        t3 = time.time()
        print("SVD done in", round(t3-t2,2), "seconds")
        V = svd.components_

        self.Us = Us
        self.V = V

        # decomp = NMF(n_components=2, init="random")




    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:

        self.users_emb = self.Us.astype(np.float32)
        self.tracks_emb = self.V.T.astype(np.float32)

        print("Computing similarities")
        start = time.time()
        # cos_mat = cossim(users_emb, tracks_emb).cpu().detach().numpy()
        cos_mat = cosine_similarity(self.users_emb, self.tracks_emb)
        # cos_mat = self.users_emb @ self.track_emb
        stop = time.time()
        print("Similarities computed in", stop-start, "seconds")
        self.cos_mat = cos_mat

        known_tracks_array = np.array([ self.rev_track_map[i] for i in range(len(self.track_map))])
        results = np.zeros((len(user_ids), self.top_k), dtype=int)

        known_likes = {}
        for user, grp in self.train_df.groupby("user_id"):
            known_likes[user] = set(grp["track_id"])

        overlaps = []
        horizon = 1
        with tqdm(range(cos_mat.shape[0])) as bar:
            for i in bar:
                curr_k = self.top_k * horizon + len(known_likes[int(user_ids.iloc[i])])

                parts = np.argpartition(-cos_mat[i], kth=curr_k)[:curr_k]
                cos_mat_sub = known_tracks_array[parts[np.argsort(-cos_mat[i,parts])]]
                chosen = np.zeros(self.top_k * horizon)
                j = 0
                k = 0
                overlaps.append(len(set(cos_mat_sub)&known_likes[int(user_ids.iloc[i])]) / len(known_likes[int(user_ids.iloc[i])]))
                ### TODO: overlap between found and known is small!!!
                while k < self.top_k * horizon:
                    if cos_mat_sub[j] not in known_likes[int(user_ids.iloc[i])]:
                        chosen[k] = cos_mat_sub[j]
                        k += 1
                    j += 1

                p = 1/(1+np.arange(len(chosen)))# + 1 * tracks_pop[chosen].values + 1 * tracks_artist_pop[chosen].values

                p = p / p.sum()
                subset = np.random.choice(len(chosen), size=self.top_k, p=p, replace=False)
            
                results[i] = chosen[sorted(subset)] #sort_by_order(chosen, subset)

        preds = results
        data = np.hstack([ user_ids["user_id"].values.reshape(-1, 1), preds ])
        return pd.DataFrame(data, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')



# if __name__ == "__main__":

#     dataset = ChallengeDataset()
#     train, test = dataset.get_sample_train_test()

#     model = MyModel(dataset.df_tracks, dataset.df_users)
#     model.train(train)
#     model.predict(test[["user_id"]])