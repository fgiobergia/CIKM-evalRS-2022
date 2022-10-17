import numpy as np
import pandas as pd
from reclist.abstractions import RecModel

import time
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ## Execution times
# The following are some estimates of the execution time of the various parts of the code.

# * Word2Vec training: 280 - 340 seconds, with 8 workers, 1 epoch
# * Model training (per epoch): 190s (w/ 3090) / 260s (w/ V100)
# * Predictions: ~ 75s 
# * Other misc activities: ~ 250 seconds (data preprocessing, setting up models, etc.)
# * Total time (minutes): 16.452068734169007 (3090 machine), 19.756832940928441 (V100 machine)


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self, n_epochs):
        self.bar = tqdm(n_epochs)

    def on_epoch_begin(self, model):
        self.bar.update()

    def on_epoch_end(self, model):
        self.bar.set_postfix(loss=model.get_latest_training_loss())
    
    def on_train_end(self, model):
        self.bar.close()

def get_track_rel_weight(train_df, trait):
    # trait: artist_id, track_id, user_id
    gb = train_df.groupby(trait)["user_track_count"].sum()
    ndx = gb.index.tolist()
    weights = 1/np.log(gb.values+1)
    weights = weights / weights.sum()

    mapper = dict(zip(ndx, weights))
    return train_df[trait].map(mapper.get)

def get_user_rel_weight(train_df, users_df, trait):
    # trait: gender, country
    gb = users_df.fillna("n").groupby(trait).size()
    ndx = gb.index.tolist()
    if trait == "gender":
        weights = 1/gb.values
    else:
        weights = 1/np.log(gb.values+1)
    weights = weights / weights.sum()

    mapper = dict(zip(ndx, weights))
    df_merged = train_df.merge(users_df.fillna("n"), left_on="user_id", right_index=True)
    return df_merged[trait].map(mapper.get)

class UserEncoder(nn.Module):
    def __init__(self, init):
        super().__init__()

        self.mat = nn.Parameter(torch.tensor(init))
    
    def forward(self, x):
        return self.mat[x.flatten()]

class TrackEncoder(nn.Module):
    def __init__(self, init):
        super().__init__()

        self.mat = nn.Parameter(torch.tensor(init))
    
    def forward(self, x):
        return self.mat[x.flatten()]

class ContrastiveModel(nn.Module):
    def __init__(self, users_vecs, tracks_vecs):
        super().__init__()
        self.user_enc = UserEncoder(users_vecs)
        self.track_enc = TrackEncoder(tracks_vecs)
    
    def forward(self, x_user, x_track_pos, x_track_neg, x_user_pos, x_user_neg, x_track_anchor):
        x_user = self.user_enc(x_user)
        x_user_pos = self.user_enc(x_user_pos)
        x_user_neg = self.user_enc(x_user_neg)
        x_track_pos = self.track_enc(x_track_pos)
        x_track_neg = self.track_enc(x_track_neg)
        x_track_anchor = self.track_enc(x_track_anchor)

        return x_user, x_track_pos, x_track_neg, x_user_pos, x_user_neg, x_track_anchor

def cossim(a, b):
    return (a @ b.T) / ((((a**2).sum(axis=1))**.5).reshape(-1,1) * ((b**2).sum(axis=1))**.5)

class UserTrackDataset():
    def __init__(self, X_user, X_track, users_map, tracks_map, train_df, w, device):
        assert X_user.shape[0] == X_track.shape[0]
        self.device = device

        self.X_user = torch.tensor(X_user, device=self.device)
        self.X_track = torch.tensor(X_track, device=self.device)

        # contains : user: { songs listened }
        self.listened_set = { users_map[k]: set([ tracks_map[i] for i in set(g["track_id"]) ]) for k, g in train_df.groupby("user_id") }
        self.listened = { k: torch.tensor(list(v), device=self.device) for k,v in self.listened_set.items() }

        # contains: song : { users listening }
        self.listeners = { tracks_map[k]: torch.tensor(list(set([ users_map[i] for i in set(g["user_id"]) ])), device=self.device) for k, g in train_df.groupby("track_id") }

        self.w = torch.tensor(w, device=self.device)

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
            # topk = neighbors.topk(use_top+1)
            # topk_ind = topk.indices[:,1:].numpy()
            # topk_val = (topk.values[:,1:] / topk.values[:,1:].sum(axis=1).reshape(-1,1)).numpy()
            # user_nn = [ np.random.choice(topk_ind[i], p=topk_val[i]) for i in range(topk_ind.shape[0]) ]
            user_nn = neighbors.topk(use_top+1).indices[:,1:].tolist() # take most similar neighbor
        else:
            user_nn = (-neighbors).topk(use_top).indices.tolist() # take least similar neighbor

        self.songs_pool = {}
        for k, nns in enumerate(user_nn):
            # for the k-th user, consider as negatives
            # the songs listened by `nn` but not by `k`
            # for nn in [nns]:
            for nn in nns:
                diff = self.listened_set[nn] - self.listened_set[k]
                # pick the 1st neighbor that has a non-empty intersection
                # (typically will be the 1st neighbor -- might be a subsequent
                # one if all songs are in common)
                if diff:
                    self.songs_pool[k] = torch.tensor(list(diff), device=self.device)
                    break

    def __getitem__(self, i):
        x_user = self.X_user[i]
        x_track_pos = self.X_track[i]

        sp = self.songs_pool[x_user.item()] # choose a song from pool of songs for the specific user
        j = random.randint(0, len(sp)-1)
        x_track_neg = sp[j]
        # j = random.randint(0, len(self.X_track))
        # x_track_neg = self.X_track[j]

        up = self.listeners[x_track_pos.item()]
        k = random.randint(0, len(up)-1)
        x_user_pos = up[k].reshape(1)

        up = self.listeners[x_track_neg.item()]
        k = random.randint(0, len(up)-1)
        x_user_neg = up[k]

        sp = self.listened[x_user.item()]
        k = random.randint(0, len(sp)-1)
        x_track_anchor = sp[k].reshape(1)
        
        return x_user, x_track_pos, x_track_neg, x_user_pos, x_user_neg, x_track_anchor, self.w[i]


class MyModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
        # calling get_start_method() and then set_start_method()
        # raises an exception, so we instead try to set the start
        # method immediately and if an exception raises (context
        # has already been set), we handle it quietly.
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass

        self.top_k = top_k
        self.known_tracks = list(set(tracks.index.values.tolist()))

        self.lambda1 = kwargs.get("lambda1", 2.5)
        self.lambda2 = kwargs.get("lambda2", 2.5)
        self.margin = kwargs.get("margin", .25)

        self.ns_exponent = kwargs.get("ns_exponent", .5)

        self.n_epochs = kwargs.get("n_epochs", 2)

        self.negative = kwargs.get("negative", 5)
        self.horizon = kwargs.get("horizon", 5)

        self.n_dims = kwargs.get("n_dims", 256)
        self.use_w2v = kwargs.get("use_w2v", True)

        self.use_weights = kwargs.get("use_weights", True)

        # default_coef = {'artist_id': 100000.0, 'country': 500, 'gender': 10, 'track_id': 500000.0, 'user_id': 50000.0} # 1.33
        default_coef = {'artist_id': 10000.0, 'country': 100, 'gender': 5, 'track_id': 100000.0, 'user_id': 10000.0} 

        self.coef = kwargs.get("coef", default_coef)
        self.users_df = users
        self.df_tracks = tracks
    
    def train(self, train_df: pd.DataFrame):

        self.known_likes = {}
        for user, grp in train_df.groupby("user_id"):
            self.known_likes[user] = set(grp["track_id"])

        
        if self.use_w2v:
            sentences = []
            for track_id, group in train_df.groupby("track_id"):
                albums = map(int, self.df_tracks.loc[track_id]["albums_id"][1:-1].split(", "))
                artist = int(group.iloc[0]["artist_id"])
                sentence = [
                    f'track={track_id}', f'artist={artist}', *[ f"album={i}" for i in albums ]
                ]
                # adding all users in the same sentence
                sentence = [ f'user={uid}' for uid in group["user_id"] ] + sentence
                sentences.append(sentence)

            print("Training w2v for initialization")
            n_epochs = 1
            self.sentences = sentences
            self.w2v_model = Word2Vec(self.sentences, vector_size=self.n_dims,  \
                                window=max(map(len,sentences)),  \
                                workers=8,
                                sg=1, hs=0, negative=self.negative, seed=42, \
                                min_count=1, \
                                sample=1e-5, \
                                ns_exponent=self.ns_exponent, \
                                compute_loss=True, \
                                epochs=n_epochs, callbacks=[EpochLogger(n_epochs)])
            all_words = self.w2v_model.wv.index_to_key
            self.tracks_vecs = np.array([ self.w2v_model.wv[word] for word in all_words if word.startswith("track=") ])
            self.users_vecs = np.array([ self.w2v_model.wv[word] for word in all_words if word.startswith("user=") ])
        else:
            uniq_users = [ f"user={uid}" for uid in set(train_df["user_id"]) ]
            uniq_tracks = [ f"track={uid}" for uid in set(train_df["track_id"]) ]

            k_u = len(uniq_users)**-.5
            k_t = len(uniq_tracks)**-.5
            self.users_vecs = np.random.uniform(-k_u, k_u, size=(len(uniq_users), self.n_dims)).astype(np.float32)
            self.tracks_vecs = np.random.uniform(-k_t, k_t, size=(len(uniq_tracks), self.n_dims)).astype(np.float32)

            all_words = uniq_users + uniq_tracks

        self.known_tracks = [ int(word.split("=")[1]) for word in all_words if word.startswith("track=") ]
        self.rev_track_map = np.array([ int(word.replace("track=","")) for word in all_words if word.startswith("track=") ])
        self.rev_user_map = np.array([ int(word.replace("user=","")) for word in all_words if word.startswith("user=")  ])
        self.track_map = { k: v for v, k in enumerate(self.rev_track_map )}
        self.user_map = { k: v for v, k in enumerate(self.rev_user_map )}

        batch_size = 512
        num_workers = 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("batch size", batch_size, "#epochs", self.n_epochs)

        X_users = np.array([ self.user_map[i] for i in train_df["user_id"]]).reshape(-1,1)
        X_tracks = np.array([ self.track_map[i] for i in train_df["track_id"]]).reshape(-1,1)

        weights = get_track_rel_weight(train_df, "artist_id") * self.coef["artist_id"] + \
                  get_track_rel_weight(train_df, "track_id") * self.coef["track_id"] + \
                  get_track_rel_weight(train_df, "user_id") * self.coef["user_id"] + \
                  get_user_rel_weight(train_df, self.users_df, "gender") * self.coef["gender"] + \
                  get_user_rel_weight(train_df, self.users_df, "country") * self.coef["country"]
        weights = torch.tensor(weights.values)

        ds = UserTrackDataset(X_users, X_tracks, self.user_map, self.track_map, train_df, weights.tolist(), self.device)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.cmodel = ContrastiveModel(self.users_vecs, self.tracks_vecs).to(self.device)
        opt = optim.Adam(self.cmodel.parameters(), weight_decay=0.)

        def cos_dist():
            cossm = nn.CosineSimilarity()
            def func(*args, **kwargs):
                return 1 - cossm(*args, **kwargs)
            return func

        loss_func = nn.TripletMarginWithDistanceLoss(margin=self.margin, distance_function=cos_dist(), reduction="none")
        print("Training with", len(ds), "records", len(self.user_map), "users", len(self.track_map), "tracks")

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            ds.update_neighbors(self.cmodel.user_enc.mat, closest=True)
            with tqdm(enumerate(dl), total=len(dl)) as bar:
                cum_loss = 0
                alpha = .8 # damp

                for i, (x_users, x_tracks_pos, x_tracks_neg, x_user_pos, x_user_neg, x_track_anchor, w) in bar:
                    opt.zero_grad()
                    anchor, track_pos, track_neg, user_pos, user_neg, track_anchor = self.cmodel(x_users, x_tracks_pos, x_tracks_neg, x_user_pos, x_user_neg, x_track_anchor)
                    loss = loss_func(anchor, track_pos, track_neg) \
                            + self.lambda1 * loss_func(anchor, user_pos, user_neg) \
                            + self.lambda2 * loss_func(track_anchor, track_pos, track_neg) \
                    
                    if self.use_weights:
                        loss *= w
                    
                    loss = loss.mean()
                    loss.backward()
                    opt.step()

                    cum_loss = cum_loss * alpha + (1-alpha) * loss.item()
                    bar.set_postfix(loss=cum_loss)


    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        users_emb = self.cmodel.user_enc.mat[[ self.user_map[i] for i in user_ids["user_id"]]]
        tracks_emb = self.cmodel.track_enc.mat

        print("Computing cosine similarities")
        start = time.time()
        cos_mat = cosine_similarity(users_emb.cpu().detach().numpy().astype(np.float32), tracks_emb.cpu().detach().numpy().astype(np.float32))
        stop = time.time()
        print("Similarities computed in", stop-start, "seconds")

        known_tracks_array = np.array(self.known_tracks)
        results = np.zeros((len(user_ids), self.top_k), dtype=int)

        with tqdm(range(cos_mat.shape[0])) as bar:
            for i in bar:
                liked_set = self.known_likes[int(user_ids.iloc[i])]
                curr_k = self.top_k * self.horizon + len(liked_set)

                parts = np.argpartition(-cos_mat[i], kth=curr_k)[:curr_k]
                cos_mat_sub = known_tracks_array[parts[np.argsort(-cos_mat[i,parts])]]

                rank = { v: k for k,v in enumerate(cos_mat_sub) }
                chosen = np.array(sorted(set(cos_mat_sub) - liked_set, key=rank.get))[:self.top_k * self.horizon]

                p = 1/(1+np.arange(len(chosen)))
                p = p / p.sum()
                subset = np.random.choice(len(chosen), size=self.top_k, p=p, replace=False)
            
                results[i] = chosen[sorted(subset)]

        data = np.hstack([ user_ids["user_id"].values.reshape(-1, 1), results ])
        return pd.DataFrame(data, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')

