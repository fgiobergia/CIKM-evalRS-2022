import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
import random

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
from tqdm import tqdm

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

class W2VModel(RecModel):

    def __init__(self, tracks: pd.DataFrame, users: pd.DataFrame, top_k : int = 100, **kwargs):
        self.top_k = top_k

        # sentences = []
        # for track_id, row in tracks.iterrows():
        #     sentence = [
        #         f'track={track_id}', f'artist={row["artist_id"]}', *[ f"album={i}" for i in map(int,row["albums_id"][1:-1].split(", ")) ]
        #     ]
        #     sentences.append(sentence)
        
        # n_epochs = 10
        # self.sentences = sentences
        # self.w2v_model = Word2Vec(self.sentences, vector_size=256,  \
        #                      window=max(map(len,sentences)),  \
        #                      workers=8,
        #                      sg=1, hs=0, negative=5, seed=42, \
        #                      min_count=1, \
        #                      # use sample param to downsample frequent words?
        #                      # (or, increase # epochs)
        #                      epochs=n_epochs, callbacks=[EpochLogger(n_epochs)])

        # self.tracks_vecs = np.array([ self.w2v_model.wv[word] for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ])
        # self.tracks_reverse_lookup = np.array([ int(word.replace("track=","")) for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ])
        # self.tracks_lookup = { k: v for v, k in enumerate(self.tracks_reverse_lookup )}

        self.known_tracks = list(set(tracks.index.values.tolist()))
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
        
        n_epochs = 1
        self.sentences = sentences
        self.w2v_model = Word2Vec(self.sentences, vector_size=256,  \
                             window=max(map(len,sentences)),  \
                             workers=8,
                             sg=1, hs=0, negative=5, seed=42, \
                             min_count=1, \
                             sample=1e-5, \
                             ns_exponent=0.25, \
                             compute_loss=True, \
                             # use sample param to downsample frequent words?
                             # (or, increase # epochs)
                             epochs=n_epochs, callbacks=[EpochLogger(n_epochs)])
        print("Trained")

        # known_tracks_set = set(train_df["track_id"])
        self.known_tracks = [ word for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ]
        self.tracks_vecs = np.array([ self.w2v_model.wv[word] for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ])
        self.users_vecs = np.array([ self.w2v_model.wv[word] for word in self.w2v_model.wv.index_to_key if word.startswith("user=") ])
        self.tracks_reverse_lookup = np.array([ int(word.replace("track=","")) for word in self.w2v_model.wv.index_to_key if word.startswith("track=") ])
        self.users_reverse_lookup = np.array([ int(word.replace("user=","")) for word in self.w2v_model.wv.index_to_key if word.startswith("user=")  ])
        self.tracks_lookup = { k: v for v, k in enumerate(self.tracks_reverse_lookup )}
        self.users_lookup = { k: v for v, k in enumerate(self.users_reverse_lookup )}
        

    # def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
    #     known_likes = {}
    #     print("starting predict()")
    #     for user, grp in self.train_df.groupby("user_id"):
    #         known_likes[user] = set(grp["track_id"])

    #     results = np.zeros((len(user_ids), self.top_k), dtype=int)

    #     for i, user in tqdm(enumerate(user_ids["user_id"]), total=len(user_ids)):
    #         token = f'user={user}'
    #         coef = 10
    #         while True:
    #             ms = [ int(x.split("=")[1]) for x, _ in self.w2v_model.wv.most_similar(token, topn=self.top_k * coef) if x.startswith("track=") ]
    #             if len(ms) < self.top_k + len(known_likes[user]):
    #                 coef *= 2
    #             else:
    #                 break
    #             #, f"available tracks: {len(ms)}, topk = {self.top_k}, known likes = {len(known_likes[user])}"

    #         chosen = []
    #         for m in ms:
    #             if m not in known_likes[user]:
    #                 chosen.append(m)
    #                 if len(chosen) == self.top_k:
    #                     break
            
    #         results[i] = chosen


            
    #     # X_users = np.array([ self.user_map[i] for i in user_ids["user_id"]]).reshape(-1,1)

    #     # bs = 1024

    #     # try:
    #     #     self.cmodel.eval()
    #     #     print("Loading user embeddings")
    #     #     users_emb = (torch.vstack( [ self.cmodel.user_enc(torch.tensor(X_users[i*bs:(i+1)*bs]).to(self.device)).detach().cpu() for i in range(X_users.shape[0]//bs+1)] )).cpu().detach().numpy()

    #     #     print("Loading tracks embeddings")
    #     #     X_tracks = np.array([ self.track_map[i] for i in self.known_tracks]).reshape(-1,1)
    #     #     tracks_emb = (torch.vstack( [ self.cmodel.track_enc(torch.tensor(X_tracks[i*bs:(i+1)*bs]).to(self.device)).detach().cpu() for i in range(X_tracks.shape[0]//bs+1)] )).cpu().detach().numpy()
    #     # finally:
    #     #     self.cmodel.train()

    #     # self.users_emb = users_emb
    #     # self.track_emb = tracks_emb

    #     # print("Computing cosine similarities")
    #     # cos_mat = cosine_similarity(users_emb, tracks_emb)
    #     # self.cos_mat = cos_mat

    #     # print("Sorting similarities")

    #     # known_tracks_array = np.array(self.known_tracks)
    #     # assert len(user_ids) == cos_mat.shape[0]
    #     # results = np.zeros((len(user_ids), self.top_k), dtype=int)

    #     # known_likes = {}
    #     # for user, grp in self.train_df.groupby("user_id"):
    #     #     known_likes[user] = set(grp["track_id"])

    #     # overlaps = []
    #     # horizon = 1
    #     # print("Predictions with", users_emb.shape[0], "users", tracks_emb.shape[0], "tracks")
    #     # with tqdm(range(cos_mat.shape[0])) as bar:
    #     #     for i in bar:
    #     #         curr_k = self.top_k + len(known_likes[int(user_ids.iloc[i])])

    #     #         parts = np.argpartition(-cos_mat[i], kth=curr_k)[:curr_k]
    #     #         cos_mat_sub = known_tracks_array[parts[np.argsort(-cos_mat[i,parts])]]
    #     #         chosen = np.zeros(self.top_k * horizon)
    #     #         j = 0
    #     #         k = 0
    #     #         overlaps.append(len(set(cos_mat_sub)&known_likes[int(user_ids.iloc[i])]) / len(known_likes[int(user_ids.iloc[i])]))
    #     #         ### TODO: overlap between found and known is small!!!
    #     #         while k < self.top_k * horizon:
    #     #             if cos_mat_sub[j] not in known_likes[int(user_ids.iloc[i])]:
    #     #                 chosen[k] = cos_mat_sub[j]
    #     #                 k += 1
    #     #             j += 1

    #     #         p = 1/(1+np.arange(len(chosen)))# + 1 * tracks_pop[chosen].values + 1 * tracks_artist_pop[chosen].values

    #     #         p = p / p.sum()
    #     #         subset = np.random.choice(len(chosen), size=self.top_k, p=p, replace=False)
            
    #     #         results[i] = chosen[sorted(subset)] #sort_by_order(chosen, subset)
            
    #     #     print("Overlaps report")
    #     #     print("mean", np.mean(overlaps))
    #     #     print("std", np.std(overlaps))
    #     #     print("max", np.max(overlaps), "min", np.min(overlaps))
    #     preds = results
    #     data = np.hstack([ user_ids["user_id"].values.reshape(-1, 1), preds ])
    #     return pd.DataFrame(data, columns=['user_id', *[str(i) for i in range(self.top_k)]]).set_index('user_id')

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:

        users_emb = self.users_vecs.astype(np.float32)
        tracks_emb = self.tracks_vecs.astype(np.float32)

        print("Computing cosine similarities")
        start = time.time()
        # cos_mat = cossim(users_emb, tracks_emb).cpu().detach().numpy()
        cos_mat = cosine_similarity(users_emb, tracks_emb)
        stop = time.time()
        print("Similarities computed in", stop-start, "seconds")

        known_tracks_array = np.array(self.known_tracks)
        assert len(user_ids) == cos_mat.shape[0]
        results = np.zeros((len(user_ids), self.top_k), dtype=int)

        known_likes = {}
        for user, grp in self.train_df.groupby("user_id"):
            known_likes[user] = set(grp["track_id"])

        # overlaps = []
        horizon = 5
        with tqdm(range(cos_mat.shape[0])) as bar:
            for i in bar:

                user_id = int(user_ids["user_id"].iloc[i])
                pos = self.users_lookup[user_id]

                curr_k = self.top_k * horizon + len(known_likes[user_id])

                parts = np.argpartition(-cos_mat[pos], kth=curr_k)[:curr_k]
                cos_mat_sub = [ int(x.split("=")[1]) for x in known_tracks_array[parts[np.argsort(-cos_mat[pos,parts])]] ]

                rank = { v: k for k,v in enumerate(cos_mat_sub) }
                chosen = np.array(sorted(set(cos_mat_sub) - known_likes[user_id], key=rank.get))[:self.top_k * horizon]

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

