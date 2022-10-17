import os
import json
from datetime import datetime
from dotenv import load_dotenv

from sklearn.model_selection import ParameterGrid

# # import env variables from file
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
    from evaluation.EvalRSRecList import MyEvalRSRecList
    # from submission.MyModel import * 
    from submission.ModelPretrained import * 

    res = {}
    dataset = ChallengeDataset(seed=42)
    
    runner = EvalRSRunner(
        dataset=dataset,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    
    models = {
        "no user-user": MyModel(dataset.df_tracks, dataset.df_users, lambda1=0),
        "no item-item": MyModel(dataset.df_tracks, dataset.df_users, lambda2=0),
        "no user-user, item-item": MyModel(dataset.df_tracks, dataset.df_users, lambda1=0, lambda2=0),
        "no w2v": MyModel(dataset.df_tracks, dataset.df_users, use_w2v=False),
        "no weights": MyModel(dataset.df_tracks, dataset.df_users, use_weights=False),
    }
    for model_name, model in models.items():
        print("Testing", model_name)
        score, agg_res = runner.evaluate(model=model, upload=False, custom_RecList=MyEvalRSRecList)
        print("FINISHED", model_name, score)
        res[model_name] = score
        with open("ablation-results-2.json", "w") as f:
            json.dump(res, f)
