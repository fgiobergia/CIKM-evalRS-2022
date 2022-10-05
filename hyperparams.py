import os
import json
from datetime import datetime
from dotenv import load_dotenv

from sklearn.model_selection import ParameterGrid

# import env variables from file
load_dotenv('upload.env', verbose=True)

# variables for the submission
EMAIL = os.getenv('EMAIL')  # the e-mail you used to sign up
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')  # you received it in your e-mail
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')  # you received it in your e-mail
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # you received it in your e-mail
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')  # you received it in your e-mail


# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import the basic classes
    from evaluation.EvalRSRunner import EvalRSRunner
    from evaluation.EvalRSRunner import ChallengeDataset
    from submission.MyModel import * 

    dataset = ChallengeDataset(seed=112314545, num_folds=1)
    runner = EvalRSRunner(
        dataset=dataset,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    
    # # lambda1 = [ .5, 1., 5., 10]
    # # lambda2 = [ .5, 1., 5., 10]
    # # margin = [ 0., .5, 1., 3 ]
    # lambda1 = [ .5, 1., 2. ]
    # lambda2 = [ .5, 1., 2. ]
    # margin = [.25, .5, .75 ]
    # with open("out.json") as f:
    #     res = json.load(f)
    # # res = {}

    # cont = False
    # for l1 in lambda1:
    #     for l2 in lambda2:
    #         for m in margin:
    #             if l1 == 2. and l2 == 2. and m == .75:
    #                 cont = True
    #             if not cont:
    #                 continue
    #             print("Config", l1, l2, m)
    #             my_model = MyModel(dataset.df_tracks, dataset.df_users, lambda1=l1, lambda2=l2,margin=m)
    #             score, agg_res = runner.evaluate(model=my_model)
    #             print(f"lambda1 = {l1}, lambda2 = {l2}, margin={m}, result={score}")
    #             res[f"l1={l1} l2={l2} m={m}"] = score, agg_res
    #             with open("out.json", "w") as f:
    #                 json.dump(res, f)
    # print(res)


    # params = [[0.5, 1.0, 0.5],
    #           [0.5, 2.0, 0.25],
    #           [0.5, 2.0, 0.75],
    #           [1.0, 0.5, 0.75],
    #           [1.0, 0.5, 0.25],
    #           [2.0, 1.0, 0.25],
    #           [2.0, 2.0, 0.5],
    #           [1.0, 2.0, 0.25],
    #           [2.0, 1.0, 0.75],
    #           [1.0, 1.0, 0.25],
    #           [2.0, 0.5, 0.25],
    #           [0.5, 2.0, 0.5],
    #           [1.0, 2.0, 0.5],
    #           [2.0, 2.0, 0.25]]
    # res = {}
    # for l1, l2, m in params[::-1]:
    #     my_model = MyModel(dataset.df_tracks, dataset.df_users, lambda1=l1, lambda2=l2,margin=m)
    #     score, agg_res = runner.evaluate(model=my_model)
    #     print(f"lambda1 = {l1}, lambda2 = {l2}, margin={m}, result={score}")
    #     res[f"l1={l1} l2={l2} m={m}"] = score, agg_res
    #     with open("out-2-9876.json", "w") as f:
    #         json.dump(res, f)
    # print(res)


    # res = {}

    # configs = list(ParameterGrid({
    #     "artist_id": [ 0, 1e4, 5e4, 1e5 ],
    #     "track_id": [ 0, 1e5, 5e5, 1e6 ],
    #     "gender": [ 0, 1, 5, 10 ],
    #     "country": [0, 100, 500, 1000 ],
    #     "user_id": [ 0, 1e4, 5e4, 1e5]
    # }))
    configs = list(ParameterGrid({
        "lambda1": np.linspace(0, 10, 5),
        "lambda2": np.linspace(0, 10, 5),
        "margin": np.linspace(0.1, .9, 5),
    }))

    np.random.seed(9493)
    choices = np.random.choice(configs, size=50, replace=False)

    # for config in choices:
    #     my_model = MyModel(dataset.df_tracks, dataset.df_users, coef=config)
    #     score, agg_res = runner.evaluate(model=my_model)
    #     print(f"{config}, result={score}")
    #     res[f"{config}"] = score, agg_res
    #     with open("out-coef-123456.json", "w") as f:
    #         json.dump(res, f)

    res = {}
    # choices = [{'artist_id': 10000.0, 'country': 100, 'gender': 5, 'track_id': 100000.0, 'user_id': 10000.0},
    #             {'artist_id': 0, 'country': 0, 'gender': 1, 'track_id': 1000000.0, 'user_id': 100000.0},
    #             {'artist_id': 50000.0, 'country': 100, 'gender': 5, 'track_id': 500000.0, 'user_id': 50000.0},
    #             {'artist_id': 50000.0, 'country': 1000, 'gender': 1, 'track_id': 1000000.0, 'user_id': 50000.0},
    #             {'artist_id': 10000.0, 'country': 0, 'gender': 1, 'track_id': 500000.0, 'user_id': 0},
    #             {'artist_id': 10000.0, 'country': 500, 'gender': 5, 'track_id': 500000.0, 'user_id': 10000.0},
    #             {'artist_id': 100000.0, 'country': 100, 'gender': 10, 'track_id': 100000.0, 'user_id': 50000.0},
    #             {'artist_id': 50000.0, 'country': 500, 'gender': 10, 'track_id': 100000.0, 'user_id': 100000.0},
    #             {'artist_id': 100000.0, 'country': 1000, 'gender': 10, 'track_id': 0, 'user_id': 10000.0},
    #             {'artist_id': 10000.0, 'country': 0, 'gender': 5, 'track_id': 0, 'user_id': 50000.0}]

    # choices = [{'artist_id': 10000.0, 'country': 500, 'gender': 5, 'track_id': 500000.0, 'user_id': 10000.0},
    #            {'artist_id': 0, 'country': 0, 'gender': 1, 'track_id': 1000000.0, 'user_id': 100000.0},
    #            {'artist_id': 10000.0, 'country': 100, 'gender': 5, 'track_id': 100000.0, 'user_id': 10000.0},
    #            {'artist_id': 50000.0, 'country': 500, 'gender': 10, 'track_id': 100000.0, 'user_id': 100000.0}]


    # choices = [{'artist_id': 10000.0, 'country': 0, 'gender': 0, 'track_id': 0, 'user_id': 50000.0},
    #     {'artist_id': 100000.0, 'country': 500, 'gender': 10, 'track_id': 0, 'user_id': 50000.0},
    #     {'artist_id': 10000.0, 'country': 0, 'gender': 10, 'track_id': 100000.0, 'user_id': 0},
    #     {'artist_id': 0, 'country': 100, 'gender': 1, 'track_id': 100000.0, 'user_id': 50000.0},
    #     {'artist_id': 10000.0, 'country': 500, 'gender': 5, 'track_id': 100000.0, 'user_id': 10000.0},
    #     {'artist_id': 10000.0, 'country': 1000, 'gender': 1, 'track_id': 0, 'user_id': 0},
    #     {'artist_id': 50000.0, 'country': 0, 'gender': 10, 'track_id': 1000000.0, 'user_id': 0},
    #     {'artist_id': 10000.0, 'country': 1000, 'gender': 1, 'track_id': 0, 'user_id': 50000.0},
    #     {'artist_id': 100000.0, 'country': 1000, 'gender': 5, 'track_id': 500000.0, 'user_id': 10000.0},
    #     {'artist_id': 10000.0, 'country': 100, 'gender': 10, 'track_id': 0, 'user_id': 0},
    #     {'artist_id': 10000.0, 'country': 0, 'gender': 5, 'track_id': 100000.0, 'user_id': 100000.0},
    #     {'artist_id': 0, 'country': 500, 'gender': 1, 'track_id': 500000.0, 'user_id': 0},
    #     {'artist_id': 0, 'country': 0, 'gender': 1, 'track_id': 100000.0, 'user_id': 50000.0},
    #     {'artist_id': 50000.0, 'country': 1000, 'gender': 1, 'track_id': 1000000.0, 'user_id': 50000.0},
    #     {'artist_id': 10000.0, 'country': 100, 'gender': 5, 'track_id': 1000000.0, 'user_id': 50000.0}]
    for config in choices:
        my_model = MyModel(dataset.df_tracks, dataset.df_users, **config)
        score, agg_res = runner.evaluate(model=my_model)
        print(f"{config}, result={score}")
        res[f"{config}"] = score, agg_res
        with open("out-llm-112314545.json", "w") as f:
            json.dump(res, f)
        del my_model
