import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
import random

# from gensim.models.callback import CallbackAny2Vec
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def get_artists_weight(train_df, n_bins=10):
    gb = train_df.groupby("track_id").size()
    artists = gb.index.tolist()
    counts = gb.tolist()

    bins = np.logspace(np.log10(min(counts)), np.log10(max(counts)), n_bins, base=10)
    bins[-1]+=1

    weights = np.histogram(counts, bins=bins)[0]
    weights = weights/weights.sum()

    mapper = dict(zip(artists, weights[np.digitize(counts, bins=bins)-1]))
    return train_df["track_id"].map(mapper.get)

def get_gender_weight(train_df, users_df):
    gb = users_df.fillna("n").groupby("gender").size()
    genders = gb.index.tolist()
    weights = 1/gb.values
    weights = weights / weights.sum()
    gender_weights = dict(zip(genders, weights))
    users_lookup = dict(zip(users_df.index, users_df["gender"].fillna("n").map(gender_weights.get)))
    return train_df["user_id"].map(users_lookup.get)

# class EpochLogger(CallbackAny2Vec):
#     '''Callback to log information about training'''
#     def __init__(self, n_epochs):
#         self.bar = tqdm(n_epochs)

#     def on_epoch_begin(self, model):
#         self.bar.update()

#     def on_epoch_end(self, model):
#         self.bar.set_postfix(loss=model.get_latest_training_loss())
    
#     def on_train_end(self, model):
#         self.bar.close()

class UserEncoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.fc = nn.Linear(in_size, out_size)
    
    def forward(self, x):
        x = self.fc(x)
        return x


class TrackEncoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.fc = nn.Linear(in_size, out_size)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, user_size, track_size, n_dim):
        super().__init__()
        self.user_enc = UserEncoder(user_size, n_dim)
        self.track_enc = TrackEncoder(track_size, n_dim)

        self.cos = nn.CosineSimilarity()
    
    def forward(self, x_user, x_track_pos, x_track_neg):
        x_user = self.user_enc(x_user)
        x_track_pos = self.track_enc(x_track_pos)
        x_track_neg = self.track_enc(x_track_neg)

        pos_cos = self.cos(x_user, x_track_pos)
        neg_cos = self.cos(x_user, x_track_neg)

        return neg_cos - pos_cos

class UserTrackDataset():
    def __init__(self, X_user, X_track, w, device=None):
        assert X_user.shape[0] == X_track.shape[0]

        self.X_user = X_user
        self.X_track = X_track
        self.w = w

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def __len__(self):
        return self.X_user.shape[0]
    
    def _tensorify(self, x):
        return torch.tensor(x.todense(), device=self.device).flatten()

    def __getitem__(self, i):
        x_user = self._tensorify(self.X_user[i])
        x_track_pos = self._tensorify(self.X_track[i])
        
        j = random.randint(0, len(self)-1)
        
        # The approach below searches for a guaranteed negative sample
        # (i.e. make sure that the sample actually belongs to some other user).
        # however, it is a bit slower (1h30 vs 1h15 for 1 epoch), so we'll
        # trust that, on large scales, very few "false" negatives will be chosen
        # same_user = True
        # while same_user:
        #     j = random.randint(0, len(self)-1)
        #     if (self.X_user[j] != self.X_user[i]).todense().any():
        #         same_user = False
        
        x_track_neg = self._tensorify(self.X_track[j])
        return x_user, x_track_pos, x_track_neg, torch.tensor(self.w[i], device=self.device)

class MyModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
        try:
            torch.multiprocessing.set_start_method('spawn')# good solution !!!!
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
        n_epochs = 1
        shared_emb_dim = 128
        num_workers = 4

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ohe_users = OneHotEncoder(dtype=np.float32)
        self.ohe_tracks = OneHotEncoder(dtype=np.float32)

        X_users = self.ohe_users.fit_transform(train_df["user_id"].values.reshape(-1,1))
        X_tracks = self.ohe_tracks.fit_transform(train_df["track_id"].values.reshape(-1,1))

        # artists_weights = get_artists_weight(train_df)
        gender_weights = get_gender_weight(train_df, self.users_df)
        ds = UserTrackDataset(X_users, X_tracks, gender_weights.tolist())
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.cmodel = ContrastiveModel(X_users.shape[1], X_tracks.shape[1], shared_emb_dim).to(self.device)
        opt = optim.Adam(self.cmodel.parameters())

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            with tqdm(enumerate(dl), total=len(dl)) as bar:
                cum_loss = 0
                alpha = .8 # damp
                for i, (x_users, x_tracks_pos, x_tracks_neg, w) in bar:
                    # if i == 50:
                    #     return
                    opt.zero_grad()
                    loss = (w * self.cmodel(x_users, x_tracks_pos, x_tracks_neg)).mean()
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
        
        X_users = self.ohe_users.transform(user_ids["user_id"].values.reshape(-1,1))
        bs = 1024

        try:
            self.cmodel.eval()
            print("Loading user embeddings")
            users_emb = (torch.vstack( [ self.cmodel.user_enc(torch.tensor(X_users[i*bs:(i+1)*bs].todense()).to(self.device)).detach().cpu() for i in range(X_users.shape[0]//bs+1)] )).cpu().detach().numpy()

            print("Loading tracks embeddings")
            tracks_list = np.array(self.known_tracks).reshape(-1,1)
            X_tracks = self.ohe_tracks.transform(tracks_list)
            tracks_emb = (torch.vstack( [ self.cmodel.track_enc(torch.tensor(X_tracks[i*bs:(i+1)*bs].todense()).to(self.device)).detach().cpu() for i in range(X_tracks.shape[0]//bs+1)] )).cpu().detach().numpy()
        finally:
            self.cmodel.train()

        self.users_emb = users_emb
        self.track_emb = tracks_emb

        print("Computing cosine similarities")
        cos_mat = cosine_similarity(users_emb, tracks_emb)
        self.cos_mat = cos_mat

        print("Sorting similarities")

        include_known_tracks = False

        known_tracks_array = np.array(self.known_tracks)
        assert len(user_ids) == cos_mat.shape[0]
        results = np.zeros((len(user_ids), self.top_k), dtype=int)

        if include_known_tracks:
            bs = 8
            with tqdm(range(cos_mat.shape[0] // bs + 1)) as bar:
                for i in bar:
                    cos_mat_sub = np.argsort(-cos_mat[i*bs:(i+1)*bs], axis=1)[:, :self.top_k]
                    results[i*bs:(i+1)*bs] = cos_mat_sub
            preds = known_tracks_array[results]
        else:
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

