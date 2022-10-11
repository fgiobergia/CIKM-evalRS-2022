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
    # from submission.MyModel import * 
    from submission.ModelPretrained import * 

    res = {}
    for repeats in range(2): #[9987456, 1122334455]: # [86754]
        dataset = ChallengeDataset()#seed=seed, num_folds=1)
        runner = EvalRSRunner(
            dataset=dataset,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            participant_id=PARTICIPANT_ID,
            bucket_name=BUCKET_NAME,
            email=EMAIL
            )
        
        try:
            model_name = "w2v, ns=0.5"
            print(model_name)
            my_model = MyModel(dataset.df_tracks, dataset.df_users, ns_exponent=.5)
            score, agg_res = runner.evaluate(model=my_model)
            print("FINISHED", model_name, score)
            res[(model_name,repeats)] = score
            with open("res.json", "w") as f:
                json.dump(res, f)

        except:
            print("Exception trying", model_name, "continuing")

        try:
            model_name = "w2v, n_dims=128, ns=0.5"
            print(model_name)
            my_model = MyModel(dataset.df_tracks, dataset.df_users, n_dims=128)
            score, agg_res = runner.evaluate(model=my_model)
            print("FINISHED", model_name, score)
            res[(model_name,repeats)] = score
            with open("res.json", "w") as f:
                json.dump(res, f)

        except:
            print("Exception trying", model_name, "continuing")

        try:
            model_name = "w2v, ns=0.6"
            print(model_name)
            my_model = MyModel(dataset.df_tracks, dataset.df_users, ns_exponent=.6)
            score, agg_res = runner.evaluate(model=my_model)
            print("FINISHED", model_name, score)
            res[(model_name,repeats)] = score
            with open("res.json", "w") as f:
                json.dump(res, f)

        except:
            print("Exception trying", model_name, "continuing")

        try:
            model_name = "no w2v"
            print(model_name)
            my_model = MyModel(dataset.df_tracks, dataset.df_users, use_w2v=False)
            score, agg_res = runner.evaluate(model=my_model)
            print("FINISHED", model_name, score)
            res[(model_name,repeats)] = score
            with open("res.json", "w") as f:
                json.dump(res, f)

        except:
            print("Exception trying", model_name, "continuing")

        # outfile = f"out-llm-{seed}.json"
        # with open(outfile) as f:
        #     res = json.load(f)
        # for config in choices:
        #     key = f"{config}"
        #     if key in res:
        #         print("skipping", key)
        #         continue
        #     my_model = MyModel(dataset.df_tracks, dataset.df_users, **config)
        #     # my_model = MyModel(dataset.df_tracks, dataset.df_users, 100, coef=config)
        #     score, agg_res = runner.evaluate(model=my_model)
        #     print(f"{config}, result={score}")
        #     res[key] = score, agg_res
        #     with open(outfile, "w") as f:
        #         json.dump(res, f)
        #     del my_model
        # print(f"Stored in {outfile}")