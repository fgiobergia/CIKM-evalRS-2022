# from evaluation.EvalRSRunner import ChallengeDataset
# dataset = ChallengeDataset()
# train, test = dataset.get_sample_train_test()


from collections import Counter
from functools import reduce 
from tqdm import tqdm
from scipy.sparse import csr_matrix

train_df = train
user_map = { k: v for v, k in enumerate(list(set(train_df["user_id"]))) }
rev_user_map = { k: v for v,k in user_map.items() }
track_map = { k: v for v, k in enumerate(list(set(train_df["track_id"]))) }
rev_track_map = { k: v for v,k in track_map.items() }

# user => songs they listened
listened_set = { user_map[k]: set([ track_map[i] for i in set(g["track_id"]) ]) for k, g in train_df.groupby("user_id") }
# song => users who listened to them
listeners_set = { track_map[k]: set([ user_map[i] for i in set(g["user_id"]) ]) for k, g in train_df.groupby("track_id") }

# for track in tqdm(listeners_set):
#     c = Counter([ b for a in [ listened_set[u] - {track} for u in listeners_set[track] ] for b in a ])

user_overlaps = {}
processed = set()
for user in tqdm(listened_set):
    processed.add(user)
    c = Counter([ b for a in [ listeners_set[t] - processed for t in listened_set[user] ] for b in a ])
    for k,v in c.items():
        # ind_col.extend([user,k])
        # ind_row.extend([k,user])
        # data.extend([ v, v ])
        user_overlaps[user,k] = v
        user_overlaps[k,user] = v
# mat = csr_matrix((data, (ind_row, ind_col)), shape=(len(user_map), len(user_map)))