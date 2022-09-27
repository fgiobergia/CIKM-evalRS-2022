import numpy as np
import pandas as pd
from reclist.abstractions import RecModel

import time
import random

# from gensim.models.callback import CallbackAny2Vec
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

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

def augment_df(train):
    mapper = {}
    for _, rows in train.groupby("artist_id"):
        vals = rows.groupby(rows["track_id"]).size().index.tolist()
        for v in vals:
            mapper[v] = vals
    # mapper now contains a dict with track_id: [ other track ids from same author ]

    new_rows_num = 1 # how many rows should be added for each existing row
    # TODO: choose users to enhance with criteria, instead of everybody

#         # TODO: if we use user_track_count in the future, 
#         # find a way of using having a meaningful value here 
#         # (e.g. avg # of listened songs)

#         # TODO: currently not weighting any song more than
#         # others -- this should be fairer towards unfrequent songs.
#         # consider whether additional weight should be put towards
#         # uncommon songs
#         # TODO: currently leaving the album untouched -- it should not be used as it is!
#         # TODO: instead of drawing from pool of author, what if we draw from pool of album? *** important test <===
    new_df = train.copy()
    new_df["track_id"] = new_df["track_id"].map(lambda x : np.random.choice(mapper[x]))
    return new_df

def cossim(a, b):
    return (a @ b.T) / ((((a**2).sum(axis=1))**.5).reshape(-1,1) * ((b**2).sum(axis=1))**.5)

class UserTrackDataset():
    def __init__(self, X_user, X_track, users_map, tracks_map, train_df, device):
        assert X_user.shape[0] == X_track.shape[0]
        self.device = device

        self.X_user = torch.tensor(X_user, device=self.device)
        self.X_track = torch.tensor(X_track, device=self.device)
#
        # self.rev_users_map = rev_users_map

        self.listened = { users_map[k]: set([ tracks_map[i] for i in set(g["track_id"]) ]) for k, g in train_df.groupby("user_id") }


    def __len__(self):
        return self.X_user.shape[0]
    
    def update_neighbors(self, mat, closest=True):
        use_top = 5
        with torch.no_grad():
            W = mat.cpu()
            neighbors = cossim(W, W) # compute, for each neighbor, its NN (in terms of cosine similarity)
        # user = [ self.rev_users_map[i] for i in range(mat.shape[0]) ]
        # user_nn = neighbors.argmax(axis=1) # get closest one for each row (TODO: this can be done in a soft way -- currently taking hard neighbor)
        # taking 1: to avoid "itself" (expecting it to always be in the 1st position -- as its cosdist is 1)
        if closest:
            user_nn = neighbors.topk(use_top+1).indices[:,1:].tolist() # take most similar neighbor
        else:
            user_nn = (-neighbors).topk(use_top).indices.tolist() # take least similar neighbor

        self.songs_pool = {}
        for k, nns in enumerate(user_nn):
            # for the k-th user, consider as negatives
            # the songs listened by `nn` but not by `k`
            for nn in nns:
                diff = self.listened[nn] - self.listened[k]
                # pick the 1st neighbor that has a non-empty intersection
                # (typically will be the 1st neighbor -- might be a subsequent
                # one if all songs are in common)
                if diff:
                    self.songs_pool[k] = torch.tensor(list(diff), device=self.device)
                    break

    def __getitem__(self, i):
        x_user = self.X_user[i]
        x_track_pos = self.X_track[i]
        sp = self.songs_pool[x_user.item()] # pool of songs for the specific user
        j = random.randint(0, len(sp)-1)
        # x_track_neg = self.X_track[j]
        x_track_neg = sp[j]
        
        return x_user, x_track_pos, x_track_neg

class MyModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        self.top_k = top_k
        self.known_tracks = list(set(tracks.index.values.tolist()))
    
    def train(self, train_df: pd.DataFrame):
        # option 1: embed each user/track as a 1-hot vector

        # TODO: currently only considering songs we see during
        # training. In a future solution, we will generalize to
        # unseen songs by considering their proximity in some
        # embedding space (e.g. b/c they share the same author/album)
        self.known_tracks = list(set(train_df["track_id"].values.tolist()))
        self.train_df = train_df
        # train_df = pd.concat([ train_df, augment_df(train_df) ])
        
        batch_size = 512
        n_epochs = 3 
        shared_emb_dim = 256
        num_workers = 4
        margin = .75
        print("batch size", batch_size, "#epochs", n_epochs, "emb dim", shared_emb_dim, "margin", margin)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.user_map = { k: v for v, k in enumerate(list(set(train_df["user_id"]))) }
        self.rev_user_map = { k: v for v,k in self.user_map.items() }
        self.track_map = { k: v for v, k in enumerate(list(set(train_df["track_id"]))) }
        self.rev_track_map = { k: v for v,k in self.track_map.items() }

        X_users = np.array([ self.user_map[i] for i in train_df["user_id"]]).reshape(-1,1)
        X_tracks = np.array([ self.track_map[i] for i in train_df["track_id"]]).reshape(-1,1)

        self.X_users = X_users
        self.X_tracks = X_tracks

        ds = UserTrackDataset(X_users, X_tracks, self.user_map, self.track_map, train_df, self.device)
        self.ds = ds
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.cmodel = ContrastiveModel(len(self.user_map), len(self.track_map), shared_emb_dim).to(self.device)
        opt = optim.Adam(self.cmodel.parameters())

        def cos_dist():
            cossm = nn.CosineSimilarity()
            def func(*args, **kwargs):
                return 1 - cossm(*args, **kwargs)
            return func
        loss_func = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=cos_dist())
        print("Training with", len(ds), "records", len(self.user_map), "users", len(self.track_map), "tracks")

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            ds.update_neighbors(self.cmodel.user_enc.mat, closest=True)
            with tqdm(enumerate(dl), total=len(dl)) as bar:
                cum_loss = 0
                alpha = .8 # damp
                for i, (x_users, x_tracks_pos, x_tracks_neg) in bar:
                    opt.zero_grad()
                    anchor, pos, neg = self.cmodel(x_users, x_tracks_pos, x_tracks_neg)
                    loss = loss_func(anchor, pos, neg)
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
        
        X_users = np.array([ self.user_map[i] for i in user_ids["user_id"]]).reshape(-1,1)

        bs = 1024

        try:
            self.cmodel.eval()
            print("Loading user embeddings")
            users_emb = torch.vstack( [ self.cmodel.user_enc(torch.tensor(X_users[i*bs:(i+1)*bs]).to(self.device)) for i in range(X_users.shape[0]//bs+1)] )

            print("Loading tracks embeddings")
            X_tracks = np.array([ self.track_map[i] for i in self.known_tracks]).reshape(-1,1)
            tracks_emb = torch.vstack( [ self.cmodel.track_enc(torch.tensor(X_tracks[i*bs:(i+1)*bs]).to(self.device)) for i in range(X_tracks.shape[0]//bs+1)] )
        finally:
            self.cmodel.train()

        self.users_emb = users_emb
        self.track_emb = tracks_emb

        print("Computing cosine similarities")
        start = time.time()
        # cos_mat = cossim(users_emb, tracks_emb).cpu().detach().numpy()
        cos_mat = cosine_similarity(users_emb.cpu().detach().numpy(), tracks_emb.cpu().detach().numpy())
        stop = time.time()
        print("Similarities computed in", stop-start, "seconds")
        self.cos_mat = cos_mat

        known_tracks_array = np.array(self.known_tracks)
        assert len(user_ids) == cos_mat.shape[0]
        results = np.zeros((len(user_ids), self.top_k), dtype=int)

        known_likes = {}
        for user, grp in self.train_df.groupby("user_id"):
            known_likes[user] = set(grp["track_id"])

        overlaps = []
        horizon = 5
        print("Predictions with", users_emb.shape[0], "users", tracks_emb.shape[0], "tracks")
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
            
            print("Overlaps report")
            print("mean", np.mean(overlaps))
            print("std", np.std(overlaps))
            print("max", np.max(overlaps), "min", np.min(overlaps))
        preds = results
        data = np.hstack([ user_ids["user_id"].values.reshape(-1, 1), preds ])
        return pd.DataFrame(data, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')

