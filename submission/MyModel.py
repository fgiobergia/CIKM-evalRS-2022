import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
import random

# from gensim.models.callback import CallbackAny2Vec
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


def get_track_rel_weight(train_df, trait):
    # trait: artist_id, track_id, user_id
    gb = train_df.groupby(trait)["user_track_count"].sum()
    ndx = gb.index.tolist()
    weights = 1/np.log(gb.values+1)
    # weights = (weights - weights.min()) / (weights.max() - weights.min())
    weights = weights / weights.sum()

    mapper = dict(zip(ndx, weights))
    return train_df[trait].map(mapper.get)

def get_user_rel_weight(train_df, users_df, trait):
    # train in ["gender", "country"]
    gb = users_df.fillna("n").groupby(trait).size()
    ndx = gb.index.tolist()
    if trait == "gender":
        weights = 1/gb.values
    else:
        weights = 1/np.log(gb.values+1)
    # weights = (weights - weights.min()) / (weights.max() - weights.min())
    weights = weights / weights.sum()

    mapper = dict(zip(ndx, weights))
    df_merged = train_df.merge(users_df.fillna("n"), left_on="user_id", right_index=True)
    return df_merged[trait].map(mapper.get)

class UserEncoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        k = 1 / (in_size ** .5)
        self.mat = nn.Parameter(torch.empty((in_size, out_size)).uniform_(-k,  k))
    
    def forward(self, x):
        return self.mat[x.flatten()]


class TrackEncoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        k = 1 / (in_size ** .5)
        self.mat = nn.Parameter(torch.empty((in_size, out_size)).uniform_(-k,  k))
    
    def forward(self, x):
        return self.mat[x.flatten()]

class ContrastiveModel(nn.Module):
    def __init__(self, user_size, track_size, n_dim):
        super().__init__()
        self.user_enc = UserEncoder(user_size, n_dim)
        self.track_enc = TrackEncoder(track_size, n_dim)
    
    def forward(self, x_user, x_track_pos, x_track_neg):
        x_user = self.user_enc(x_user)
        x_track_pos = self.track_enc(x_track_pos)
        x_track_neg = self.track_enc(x_track_neg)

        return x_user, x_track_pos, x_track_neg

        # pos_cos = self.cos(x_user, x_track_pos)
        # neg_cos = self.cos(x_user, x_track_neg)

class UserTrackDataset():
    def __init__(self, X_user, X_track, X_plays, w, device=None):
        assert X_user.shape[0] == X_track.shape[0]
        self.device = device

        self.X_user = torch.tensor(X_user, device=self.device)
        self.X_track = torch.tensor(X_track, device=self.device)
        self.X_plays = torch.tensor(X_plays, device=self.device)
        self.w = torch.tensor(w, device=self.device)

    def __len__(self):
        return self.X_user.shape[0]

    def __getitem__(self, i):
        x_user = self.X_user[i]
        x_track_pos = self.X_track[i]
        j = random.randint(0, len(self)-1)
        x_track_neg = self.X_track[j]
        
        
        # The approach below searches for a guaranteed negative sample
        # (i.e. make sure that the sample actually belongs to some other user).
        # however, it is a bit slower (1h30 vs 1h15 for 1 epoch), so we'll
        # trust that, on large scales, very few "false" negatives will be chosen
        # same_user = True
        # while same_user:
        #     j = random.randint(0, len(self)-1)
        #     if (self.X_user[j] != self.X_user[i]).todense().any():
        #         same_user = False
        return x_user, x_track_pos, x_track_neg, self.w[i], self.X_plays[i]

class MyModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        self.top_k = top_k

        # The stuff below may be needed if running word2vec or other embedding
        # algorithms which require words (tokens) and not numerical values
        # (i.e. do some discretization)
        # n_bins = 10

        # users = users.reset_index()
        # tracks = tracks.reset_index()
        
        # # convert user features that are numerical into bins
        # for col in users.select_dtypes(np.float64):
        #     users[col] = pd.qcut(users[col], q=n_bins, duplicates="drop").astype(str)

        # for col in users:
        #     users[col] = users[col].map(lambda x: f"{col}={x}")

        # tracks.drop(columns=["track", "artist", "albums_id", "albums"], inplace=True)

        # for col in tracks:
        #     tracks[col] = tracks[col].map(lambda x: f"{col}={x}")

        self.known_tracks = list(set(tracks.index.values.tolist()))
        self.users_df = users


    
    def train(self, train_df: pd.DataFrame):
        # option 1: embed each user/track as a 1-hot vector

        # TODO: currently only considering songs we see during
        # training. In a future solution, we will generalize to
        # unseen songs by considering their proximity in some
        # embedding space (e.g. b/c they share the same author/album)
        self.known_tracks = list(set(train_df["track_id"].values.tolist()))
        self.train_df = train_df
        
        batch_size = 512
        n_epochs = 2
        shared_emb_dim = 256
        num_workers = 4
        margin = .75
        print("batch size", batch_size, "#epochs", n_epochs, "emb dim", shared_emb_dim, "margin", margin)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.ohe_users = OneHotEncoder(dtype=np.float32)
        # self.ohe_tracks = OneHotEncoder(dtype=np.float32)

        # X_users = self.ohe_users.fit_transform(train_df["user_id"].values.reshape(-1,1))
        # X_tracks = self.ohe_tracks.fit_transform(train_df["track_id"].values.reshape(-1,1))

        self.user_map = { k: v for v, k in enumerate(list(set(train_df["user_id"]))) }
        self.rev_user_map = { k: v for v,k in self.user_map.items() }
        self.track_map = { k: v for v, k in enumerate(list(set(train_df["track_id"]))) }
        self.rev_track_map = { k: v for v,k in self.track_map.items() }

        # X_users = train_df["user_id"].values.reshape(-1,1)
        # X_tracks = train_df["track_id"].values.reshape(-1,1)
        X_users = np.array([ self.user_map[i] for i in train_df["user_id"]]).reshape(-1,1)
        X_tracks = np.array([ self.track_map[i] for i in train_df["track_id"]]).reshape(-1,1)
        X_plays = np.log(train_df["user_track_count"].values.reshape(-1,1))
        X_plays = X_plays/X_plays.max()

        self.X_users = X_users
        self.X_tracks = X_tracks

        l = {
            "artist_id": 1e4,
            "track_id": 1e5,
            "gender": 1.,
            "country": 100.,
            "user_id": 1e4,
        }
        print(l)
        weights = get_track_rel_weight(train_df, "artist_id") * l["artist_id"] + \
                  get_track_rel_weight(train_df, "track_id") * l["track_id"] + \
                  get_track_rel_weight(train_df, "user_id") * l["user_id"] + \
                  get_user_rel_weight(train_df, self.users_df, "gender") * l["gender"] + \
                  get_user_rel_weight(train_df, self.users_df, "country") * l["country"]
        weights = torch.tensor(weights.values)


        ds = UserTrackDataset(X_users, X_tracks, X_plays, weights.tolist(), self.device)
        # wrs = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.cmodel = ContrastiveModel(len(self.user_map), len(self.track_map), shared_emb_dim).to(self.device)
        opt = optim.Adam(self.cmodel.parameters())

        def cos_dist():
            cossim = nn.CosineSimilarity()
            def func(*args, **kwargs):
                return 1 - cossim(*args, **kwargs)
            return func
        loss_func = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=cos_dist(), reduction="none")

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            with tqdm(enumerate(dl), total=len(dl)) as bar:
                cum_loss = 0
                alpha = .8 # damp
                for i, (x_users, x_tracks_pos, x_tracks_neg, w, w_p) in bar:
                    opt.zero_grad()
                    anchor, pos, neg = self.cmodel(x_users, x_tracks_pos, x_tracks_neg)
                    loss = (w * loss_func(anchor, pos, neg)).mean()
                    loss.backward()
                    opt.step()

                    cum_loss = cum_loss * alpha + (1-alpha) * loss.item()
                    bar.set_postfix(loss=cum_loss)


    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        """
        
        This function takes as input all the users that we want to predict the top-k items for, and 
        returns all the predicted songs.

        While in this example is just a random generator, the same logic in your implementation 
        would allow for batch predictions of all the target data points.
        
        """
        
        # X_users = self.ohe_users.transform(user_ids["user_id"].values.reshape(-1,1))

        X_users = np.array([ self.user_map[i] for i in user_ids["user_id"]]).reshape(-1,1)
        # X_tracks = np.array([ self.track_map[i] for i in train_df["track_id"]]).reshape(-1,1)

        bs = 1024

        try:
            self.cmodel.eval()
            print("Loading user embeddings")
            users_emb = (torch.vstack( [ self.cmodel.user_enc(torch.tensor(X_users[i*bs:(i+1)*bs]).to(self.device)).detach().cpu() for i in range(X_users.shape[0]//bs+1)] )).cpu().detach().numpy()

            print("Loading tracks embeddings")
            # tracks_list = np.array(self.known_tracks).reshape(-1,1)
            X_tracks = np.array([ self.track_map[i] for i in self.known_tracks]).reshape(-1,1)
            tracks_emb = (torch.vstack( [ self.cmodel.track_enc(torch.tensor(X_tracks[i*bs:(i+1)*bs]).to(self.device)).detach().cpu() for i in range(X_tracks.shape[0]//bs+1)] )).cpu().detach().numpy()
        finally:
            self.cmodel.train()

        self.users_emb = users_emb
        self.track_emb = tracks_emb

        print("Computing cosine similarities")
        cos_mat = cosine_similarity(users_emb, tracks_emb)
        self.cos_mat = cos_mat

        print("Sorting similarities")

        known_tracks_array = np.array(self.known_tracks)
        assert len(user_ids) == cos_mat.shape[0]
        results = np.zeros((len(user_ids), self.top_k), dtype=int)

        known_likes = {}
        for user, grp in self.train_df.groupby("user_id"):
            known_likes[user] = set(grp["track_id"])

        with tqdm(range(cos_mat.shape[0])) as bar:
            for i in bar:
                curr_k = self.top_k + len(known_likes[int(user_ids.iloc[i])])

                parts = np.argpartition(-cos_mat[i], kth=curr_k)[:curr_k]
                cos_mat_sub = known_tracks_array[parts[np.argsort(-cos_mat[i,parts])]]
                chosen = []
                j = 0
                while len(chosen) < self.top_k:
                    if cos_mat_sub[j] not in known_likes[int(user_ids.iloc[i])]:
                        chosen.append(cos_mat_sub[j])
                    j += 1
                results[i] = chosen
        preds = results
        data = np.hstack([ user_ids["user_id"].values.reshape(-1, 1), preds ])
        return pd.DataFrame(data, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')

