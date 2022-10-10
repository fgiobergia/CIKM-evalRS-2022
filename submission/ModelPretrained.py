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
from torch.utils.data import DataLoader, WeightedRandomSampler

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

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
    # weights = 1/np.log(gb.values+1)
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
    def __init__(self, in_size, out_size, init=None):
        super().__init__()

        if init is None:
            k = 1 / (in_size ** .5)
            self.mat = nn.Parameter(torch.empty((in_size, out_size)).uniform_(-k,  k))
        else:
            self.mat = nn.Parameter(torch.tensor(init))
    
    def forward(self, x):
        return self.mat[x.flatten()]


class TrackEncoder(nn.Module):
    def __init__(self, in_size, out_size, init=None):
        super().__init__()

        if init is None:
            k = 1 / (in_size ** .5)
            self.mat = nn.Parameter(torch.empty((in_size, out_size)).uniform_(-k,  k))
        else:
            self.mat = nn.Parameter(torch.tensor(init))
    
    def forward(self, x):
        return self.mat[x.flatten()]

class ContrastiveModel(nn.Module):
    def __init__(self, user_size, track_size, n_dim, users_vecs=None, tracks_vecs=None):
        super().__init__()
        self.user_enc = UserEncoder(user_size, n_dim, users_vecs)
        self.track_enc = TrackEncoder(track_size, n_dim, tracks_vecs)
    
    def forward(self, x_user, x_track_pos, x_track_neg, x_user_pos, x_user_neg, x_track_anchor):
        x_user = self.user_enc(x_user)
        x_user_pos = self.user_enc(x_user_pos)
        x_user_neg = self.user_enc(x_user_neg)
        x_track_pos = self.track_enc(x_track_pos)
        x_track_neg = self.track_enc(x_track_neg)
        x_track_anchor = self.track_enc(x_track_anchor)

        return x_user, x_track_pos, x_track_neg, x_user_pos, x_user_neg, x_track_anchor

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

def cossim(a, b):
    return (a @ b.T) / ((((a**2).sum(axis=1))**.5).reshape(-1,1) * ((b**2).sum(axis=1))**.5)

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
    def __init__(self, X_user, X_track, users_map, tracks_map, train_df, X_plays, w, device):
        assert X_user.shape[0] == X_track.shape[0]
        self.device = device

        self.X_user = torch.tensor(X_user, device=self.device)
        self.X_track = torch.tensor(X_track, device=self.device)

        # contains : user: { songs listened }
        self.listened_set = { users_map[k]: set([ tracks_map[i] for i in set(g["track_id"]) ]) for k, g in train_df.groupby("user_id") }
        self.listened = { k: torch.tensor(list(v), device=self.device) for k,v in self.listened_set.items() }

        # contains: song : { users listening }
        self.listeners = { tracks_map[k]: torch.tensor(list(set([ users_map[i] for i in set(g["user_id"]) ])), device=self.device) for k, g in train_df.groupby("track_id") }

        self.X_plays = torch.tensor(X_plays, device=self.device)
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

        # sp = self.X_track # <== choose a song randomly
        sp = self.songs_pool[x_user.item()] # choose a song from pool of songs for the specific user
        j = random.randint(0, len(sp)-1)
        # x_track_neg = self.X_track[j]
        x_track_neg = sp[j]

        up = self.listeners[x_track_pos.item()]
        k = random.randint(0, len(up)-1)
        x_user_pos = up[k].reshape(1)

        up = self.listeners[x_track_neg.item()]
        k = random.randint(0, len(up)-1)
        x_user_neg = up[k]

        sp = self.listened[x_user.item()]
        k = random.randint(0, len(sp)-1)
        x_track_anchor = sp[k].reshape(1)
        
        return x_user, x_track_pos, x_track_neg, x_user_pos, x_user_neg, x_track_anchor, self.w[i], self.X_plays[i]


class MyModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
        random.seed(42)
        # torch.manual_seed(42)
        np.random.seed(42)
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        self.top_k = top_k
        self.known_tracks = list(set(tracks.index.values.tolist()))
        # self.lambda1 = kwargs.get("lambda1", 2.)
        # self.lambda2 = kwargs.get("lambda2", .5)
        # self.margin = kwargs.get("margin", .25)

        # self.lambda1 = kwargs.get("lambda1", 5.)
        # self.lambda2 = kwargs.get("lambda2", 2.5)
        # self.margin = kwargs.get("margin", .3)

        self.lambda1 = kwargs.get("lambda1", 2.5)
        self.lambda2 = kwargs.get("lambda2", 2.5)
        self.margin = kwargs.get("margin", .25)


        self.negative = kwargs.get("negative", 5)
        self.horizon = kwargs.get("horizon", 5)
        # self.lambda1 = kwargs.get("lambda1", 1.) <= best rn
        # self.lambda2 = kwargs.get("lambda2", 2.)
        # self.margin = kwargs.get("margin", .25)

        # default_coef = {
        #     "artist_id": 1e4,
        #     "track_id": 1e5,
        #     "gender": 1.,
        #     "country": 300.,
        #     "user_id": 3e4,
        # }

        # default_coef = {'artist_id': 10000.0, 'country': 500, 'gender': 5, 'track_id': 500000.0, 'user_id': 10000.0}
        # default_coef = {'artist_id': 0, 'country': 0, 'gender': 1, 'track_id': 1000000.0, 'user_id': 100000.0}
        # default_coef = {'artist_id': 10000.0, 'country': 100, 'gender': 5, 'track_id': 100000.0, 'user_id': 10000.0} # <== (1.30 - 1.31)
        # default_coef = {'artist_id': 0, 'country': 100, 'gender': 1, 'track_id': 100000.0, 'user_id': 50000.0} # <== second
        # default_coef = {'artist_id': 50000.0, 'country': 500, 'gender': 10, 'track_id': 100000.0, 'user_id': 100000.0}

        # default_coef = {'artist_id': 50000.0, 'country': 500, 'gender': 1, 'track_id': 100000.0, 'user_id': 100000.0}
        # default_coef = {'artist_id': 50000.0, 'country': 0, 'gender': 10, 'track_id': 0, 'user_id': 100000.0}
        # default_coef = {'artist_id': 50000.0, 'country': 500, 'gender': 5, 'track_id': 1000000.0, 'user_id': 50000.0} # 1.21
        # default_coef = {'artist_id': 0, 'country': 0, 'gender': 10, 'track_id': 100000.0, 'user_id': 0} # 1.16
        default_coef = {'artist_id': 100000.0, 'country': 500, 'gender': 10, 'track_id': 500000.0, 'user_id': 50000.0} # 1.33
        self.coef = kwargs.get("coef", default_coef)
        self.users_df = users
        self.df_tracks = tracks
    
    def train(self, train_df: pd.DataFrame):
        # option 1: embed each user/track as a 1-hot vector

        # TODO: currently only considering songs we see during
        # training. In a future solution, we will generalize to
        # unseen songs by considering their proximity in some
        # embedding space (e.g. b/c they share the same author/album)
        self.known_tracks = list(set(train_df["track_id"].values.tolist()))
        self.train_df = train_df
        # train_df = pd.concat([ train_df, augment_df(train_df, 4) ])

        sentences = []
        # for track_id, row in self.df_tracks.iterrows(): #dataset.df_tracks.merge(train.groupby("track_id").size().rename("count"), left_index=True, right_index=True):
        # for dataset.df_tracks.merge(train.groupby("track_id").size().rename("count"), left_index=True, right_index=True):
        
        # for _, row in tqdm(train_df.merge(self.df_tracks, right_index=True, left_on="track_id", suffixes=["_usr", ""]).iterrows(), total=len(train_df)):
        for track_id, group in train_df.groupby("track_id"):
            albums = map(int, self.df_tracks.loc[track_id]["albums_id"][1:-1].split(", "))
            artist = int(group.iloc[0]["artist_id"])
            sentence = [
                # f'track={track_id}', f'artist={row["artist_id"]}', *[ f"album={i}" for i in map(int,row["albums_id"][1:-1].split(", ")) ]
                f'track={track_id}', f'artist={artist}', *[ f"album={i}" for i in albums ]
            ]
            # adding all users in the same sentence
            sentence = [ f'user={uid}' for uid in group["user_id"] ] + sentence
            sentences.append(sentence)
        
        print("Training w2v for initialization")
        n_epochs = 1
        self.sentences = sentences
        self.w2v_model = Word2Vec(self.sentences, vector_size=256,  \
                             window=max(map(len,sentences)),  \
                             workers=8,
                             sg=1, hs=0, negative=self.negative, seed=42, \
                             min_count=1, \
                             sample=1e-5, \
                             ns_exponent=0.5, \
                             compute_loss=True, \
                             # use sample param to downsample frequent words?
                             # (or, increase # epochs)
                             epochs=n_epochs, callbacks=[EpochLogger(n_epochs)])
        print("Done.")

        # 1.16 -- 1.64? (128 vec size?)
        # self.w2v_model = Word2Vec(self.sentences, vector_size=256,  \
        #                      window=max(map(len,sentences)),  \
        #                      workers=8,
        #                      sg=1, hs=0, negative=5, seed=42, \
        #                      min_count=1, \
        #                      sample=1e-5, \
        #                      ns_exponent=0.5, \
        #                      compute_loss=True, \
        #                      # use sample param to downsample frequent words?
        #                      # (or, increase # epochs)
        #                      epochs=n_epochs, callbacks=[EpochLogger(n_epochs)])


        # known_tracks_set = set(train_df["track_id"])
        self.known_tracks = [ word for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ]
        self.tracks_vecs = np.array([ self.w2v_model.wv[word] for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ])
        self.users_vecs = np.array([ self.w2v_model.wv[word] for word in self.w2v_model.wv.index_to_key if word.startswith("user=") ])
        self.rev_track_map = np.array([ int(word.replace("track=","")) for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ])
        self.rev_user_map = np.array([ int(word.replace("user=","")) for word in self.w2v_model.wv.index_to_key if word.startswith("user=")  ])
        self.track_map = { k: v for v, k in enumerate(self.rev_track_map )}
        self.user_map = { k: v for v, k in enumerate(self.rev_user_map )}
        

        self.known_tracks = list(set(train_df["track_id"].values.tolist()))
        self.train_df = train_df
        # train_df = pd.concat([ train_df, augment_df(train_df) ])
        # train_df = pd.concat([ train_df, augment_df_user_aware(train_df, 1_000_000) ])

        #  0.58       
        # batch_size = 512
        # n_epochs = 2
        # shared_emb_dim = 256
        # num_workers = 4
        # margin = .75
        # lmbda1 = 3.
        # lmbda2 = .5
        batch_size = 512
        n_epochs = 2
        shared_emb_dim = 128
        num_workers = 4
        lmbda1 = self.lambda1
        lmbda2 = self.lambda2
        margin = self.margin
        # lmbda1 = 2.
        # lmbda2 = 2.
        # margin = 0.25

        print("batch size", batch_size, "#epochs", n_epochs, "emb dim", shared_emb_dim, "margin", margin)

        self.device = "cuda"# if torch.cuda.is_available() else "cpu"

        # self.user_map = { k: v for v, k in enumerate(list(set(train_df["user_id"]))) }
        # self.rev_user_map = { k: v for v,k in self.user_map.items() }
        # self.track_map = { k: v for v, k in enumerate(list(set(train_df["track_id"]))) }
        # self.rev_track_map = { k: v for v,k in self.track_map.items() }

        X_users = np.array([ self.user_map[i] for i in train_df["user_id"]]).reshape(-1,1)
        X_tracks = np.array([ self.track_map[i] for i in train_df["track_id"]]).reshape(-1,1)
        X_plays = np.log(train_df["user_track_count"].values.reshape(-1,1))
        X_plays = X_plays/X_plays.max()

        self.X_users = X_users
        self.X_tracks = X_tracks

        print(self.coef)
        weights = get_track_rel_weight(train_df, "artist_id") * self.coef["artist_id"] + \
                  get_track_rel_weight(train_df, "track_id") * self.coef["track_id"] + \
                  get_track_rel_weight(train_df, "user_id") * self.coef["user_id"] + \
                  get_user_rel_weight(train_df, self.users_df, "gender") * self.coef["gender"] + \
                  get_user_rel_weight(train_df, self.users_df, "country") * self.coef["country"]
        weights = torch.tensor(weights.values)


        ds = UserTrackDataset(X_users, X_tracks, self.user_map, self.track_map, train_df, X_plays, weights.tolist(), self.device)
        # wrs = WeightedRandomSampler(weights, num_samples=2*len(weights), replacement=True)
        # dl = DataLoader(ds, batch_size=batch_size, sampler=wrs, num_workers=num_workers)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.cmodel = ContrastiveModel(len(self.user_map), len(self.track_map), shared_emb_dim, self.users_vecs, self.tracks_vecs).to(self.device)
        opt = optim.Adam(self.cmodel.parameters(), weight_decay=0.)

        def cos_dist():
            cossm = nn.CosineSimilarity()
            def func(*args, **kwargs):
                return 1 - cossm(*args, **kwargs)
            return func

        loss_func = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=cos_dist(), reduction="none")
        print("Training with", len(ds), "records", len(self.user_map), "users", len(self.track_map), "tracks")

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            ds.update_neighbors(self.cmodel.user_enc.mat, closest=True)
            with tqdm(enumerate(dl), total=len(dl)) as bar:
                cum_loss = 0
                alpha = .8 # damp

                for i, (x_users, x_tracks_pos, x_tracks_neg, x_user_pos, x_user_neg, x_track_anchor, w, w_p) in bar:
                    opt.zero_grad()
                    anchor, track_pos, track_neg, user_pos, user_neg, track_anchor = self.cmodel(x_users, x_tracks_pos, x_tracks_neg, x_user_pos, x_user_neg, x_track_anchor)
                    loss = (w * (loss_func(anchor, track_pos, track_neg) \
                            + lmbda1 * loss_func(anchor, user_pos, user_neg) \
                            + lmbda2 * loss_func(track_anchor, track_pos, track_neg) \
                            # + 1 * loss_func(track_anchor, user_pos, user_neg)
                            )).mean()
                            

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

        # overlaps = []
        horizon = self.horizon
        with tqdm(range(cos_mat.shape[0])) as bar:
            for i in bar:
                curr_k = self.top_k * horizon + len(known_likes[int(user_ids.iloc[i])])

                parts = np.argpartition(-cos_mat[i], kth=curr_k)[:curr_k]
                cos_mat_sub = known_tracks_array[parts[np.argsort(-cos_mat[i,parts])]]

                rank = { v: k for k,v in enumerate(cos_mat_sub) }
                chosen = np.array(sorted(set(cos_mat_sub) - known_likes[int(user_ids.iloc[i])], key=rank.get))[:self.top_k * horizon]

                p = 1/(1+np.arange(len(chosen)))# + 1 * tracks_pop[chosen].values + 1 * tracks_artist_pop[chosen].values

                p = p / p.sum()
                subset = np.random.choice(len(chosen), size=self.top_k, p=p, replace=False)
            
                results[i] = chosen[sorted(subset)] #sort_by_order(chosen, subset)
            
            # print("Overlaps report")
            # print("mean", np.mean(overlaps))
            # print("std", np.std(overlaps))
            # print("max", np.max(overlaps), "min", np.min(overlaps))

        preds = results
        data = np.hstack([ user_ids["user_id"].values.reshape(-1, 1), preds ])
        return pd.DataFrame(data, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')

