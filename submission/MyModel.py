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

def augment_df(train, new_rows_num=1):
    # new_rows_num => how many rows should be added for each existing row
    mapper = {}
    for _, rows in train.groupby("artist_id"):
        vals = rows.groupby(rows["track_id"]).size().index.tolist()
        for v in vals:
            mapper[v] = vals
    # mapper now contains a dict with track_id: [ other track ids from same author ]

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

def augment_df_user_aware(train, new_rows):
    mapper = {}
    for _, rows in train.groupby("artist_id"):
        vals = rows.groupby(rows["track_id"]).size().index.tolist()
        for v in vals:
            mapper[v] = vals
    
    # probability proportional to # of artists the user listened to
    # (the more "focused" the user is, the more likely )
    ginis = train.groupby(["user_id","artist_id"]).size().groupby("user_id").apply(lambda grp: ((grp.values/grp.values.sum())**2).sum())
    ginis = ginis ** 3 # TODO: decide how few users we should pick (3 --> 5 ?)
    ginis = ginis / ginis.sum()

    n_users = new_rows // 5 # TODO: decide how to vary this later!
    users_pool = np.random.choice(ginis.index, p=ginis, size=n_users, replace=True)
    
    new_df = train[train["user_id"].isin(set(users_pool))].sample(new_rows, replace=True)
    new_df["track_id"] = new_df["track_id"].map(lambda x : np.random.choice(mapper[x]))
    return new_df



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
        train_df = pd.concat([ train_df, augment_df_user_aware(train_df, 1_000_000) ])
        
        batch_size = 512
        n_epochs = 4
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
        X_plays = np.log(train_df["user_track_count"].values.reshape(-1,1))
        X_plays = X_plays/X_plays.max()

        self.X_users = X_users
        self.X_tracks = X_tracks

        l = {
            "artist_id": 1e4,
            "track_id": 1e5,
            "gender": 1.,
            "country": 300.,
            "user_id": 3e4,
        }
        print(l)
        weights = get_track_rel_weight(train_df, "artist_id") * l["artist_id"] + \
                  get_track_rel_weight(train_df, "track_id") * l["track_id"] + \
                  get_track_rel_weight(train_df, "user_id") * l["user_id"] + \
                  get_user_rel_weight(train_df, self.users_df, "gender") * l["gender"] + \
                  get_user_rel_weight(train_df, self.users_df, "country") * l["country"]
        weights = torch.tensor(weights.values)


        ds = UserTrackDataset(X_users, X_tracks, X_plays, weights.tolist(), self.device)
        wrs = WeightedRandomSampler(weights, num_samples=2 * len(weights), replacement=True)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.cmodel = ContrastiveModel(len(self.user_map), len(self.track_map), shared_emb_dim).to(self.device)
        opt = optim.Adam(self.cmodel.parameters())

        def cos_dist():
            cossim = nn.CosineSimilarity()
            def func(*args, **kwargs):
                return 1 - cossim(*args, **kwargs)
            return func

        loss_func = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=cos_dist(), reduction="none")
        print("Training with", len(ds), "records", len(self.user_map), "users", len(self.track_map), "tracks")

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            with tqdm(enumerate(dl), total=len(dl)) as bar:
                cum_loss = 0
                alpha = .8 # damp
                for i, (x_users, x_tracks_pos, x_tracks_neg, w, w_p) in bar:
                    opt.zero_grad()
                    anchor, pos, neg = self.cmodel(x_users, x_tracks_pos, x_tracks_neg)
                    loss = (loss_func(anchor, pos, neg)).mean()
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
            users_emb = (torch.vstack( [ self.cmodel.user_enc(torch.tensor(X_users[i*bs:(i+1)*bs]).to(self.device)).detach().cpu() for i in range(X_users.shape[0]//bs+1)] )).cpu().detach().numpy()

            print("Loading tracks embeddings")
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

        overlaps = []
        horizon = 1
        with tqdm(range(cos_mat.shape[0])) as bar:
            for i in bar:
                curr_k = self.top_k + len(known_likes[int(user_ids.iloc[i])])

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

