"""

    This is the class containing the common methods for the evaluation of your model.
    Your own submission should NOT modify this script, but only provide your own model when running the
    evaluation.

    Please refer to the README for more details and check submission.py for a sample implementation.

"""
import os
import inspect
import hashlib
import pandas as pd
import time
import numpy as np
import json
from reclist.abstractions import RecList
from evaluation.EvalRSRecList import EvalRSRecList, EvalRSDataset
from collections import defaultdict
from evaluation.utils import download_with_progress, get_cache_directory, LFM_DATASET_PATH, decompress_zipfile, \
    upload_submission, TOP_K_CHALLENGE


class ChallengeDataset:

    def __init__(self, num_folds=4, seed: int = None, force_download: bool = False):
        # download dataset
        self.path_to_dataset = os.path.join(get_cache_directory(), 'evalrs_dataset')
        if not os.path.exists(self.path_to_dataset) or force_download:
            print("Downloading LFM dataset...")
            download_with_progress(LFM_DATASET_PATH, os.path.join(get_cache_directory(), 'evalrs_dataset.zip'))
            decompress_zipfile(os.path.join(get_cache_directory(), 'evalrs_dataset.zip'),
                               get_cache_directory())
        else:
            print("LFM dataset already downloaded. Skipping download.")

        self.path_to_events = os.path.join(self.path_to_dataset, 'evalrs_events.csv')
        self.path_to_tracks = os.path.join(self.path_to_dataset, 'evalrs_tracks.csv')
        self.path_to_users = os.path.join(self.path_to_dataset, 'evalrs_users.csv')

        assert os.path.exists(self.path_to_events)
        assert os.path.exists(self.path_to_tracks)
        assert os.path.exists(self.path_to_users)

        print("Loading dataset.")
        # TODO: Verfiy dtype do not cause overflow
        self.df_events = pd.read_csv(self.path_to_events, index_col=0, dtype='int32')
        self.df_tracks = pd.read_csv(self.path_to_tracks,
                                     dtype={
                                         'track_id': 'int32',
                                         'artist_id': 'int32'
                                     }).set_index('track_id')

        self.df_users = pd.read_csv(self.path_to_users,
                                    dtype={
                                        'user_id': 'int32',
                                        'playcount': 'int32',
                                        'country_id': 'int32',
                                        'timestamp': 'int32',
                                        'age': 'int32',
                                    }).set_index('user_id')

        print("Generating folds.")
        self.num_folds = num_folds
        self._random_state = int(time.time()) if not seed else seed
        self._test_set = self._generate_folds(self.num_folds, self._random_state)

        print("Generating dataset hashes.")
        self._events_hash = hashlib.sha256(pd.util.hash_pandas_object(self.df_events.sample(n=1000,
                                                                                            random_state=0)).values
                                           ).hexdigest()
        self._tracks_hash = hashlib.sha256(pd.util.hash_pandas_object
                                           (self.df_tracks.sample(n=1000, random_state=0)
                                            .explode(['albums', 'albums_id'])).values
                                           ).hexdigest()
        self._users_hash = hashlib.sha256(pd.util.hash_pandas_object(self.df_users.sample(n=1000,
                                                                                          random_state=0)).values
                                          ).hexdigest()

    def _generate_folds(self, num_folds: int, seed: int) -> pd.DataFrame:
        df_groupby = self.df_events.groupby(by='user_id', as_index=False)
        df_test = df_groupby.sample(n=num_folds, replace=True, random_state=seed)[['user_id', 'track_id']]
        df_test['ones'] = 1
        df_test['fold'] = df_test.groupby('user_id', as_index=False)['ones'].cumsum().values - 1
        df_test = df_test.drop('ones', axis=1)
        return df_test

    def _get_train_set(self, fold: int) -> pd.DataFrame:
        assert fold <= self._test_set['fold'].max()
        test_index = self._test_set[self._test_set['fold']==fold].index
        return self.df_events.loc[self.df_events.index.difference(test_index)]

    def _get_test_set(self, fold: int, limit: int = None) -> pd.DataFrame:
        assert fold <= self._test_set['fold'].max()
        test_set = self._test_set[self._test_set['fold'] == fold][['user_id', 'track_id']]
        if limit:
            return test_set.sample(n=limit)
        else:
            return test_set

    def get_sample_train_test(self):
        return self._get_train_set(1), self._get_test_set(1)


class EvalRSRunner:

    def __init__(self,
                 dataset: ChallengeDataset,
                 email: str = None,
                 participant_id: str = None,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 bucket_name: str = None):
        self.email = email
        self.participant_id = participant_id
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self._num_folds = None
        self._random_state = None
        self._folds = None
        self.model = None
        self.dataset = dataset

    def _test_model(self, model, fold: int, limit: int = None, custom_RecList: RecList = None) -> str:
        # use default RecList if not specified
        myRecList = custom_RecList if custom_RecList else EvalRSRecList

        test_set_df = self.dataset._get_test_set(fold=fold, limit=limit)
        x_test = test_set_df[['user_id']]
        y_test = test_set_df.set_index('user_id')

        dataset = EvalRSDataset()
        dataset.load(x_test=x_test,
                     y_test=y_test,
                     users=self.dataset.df_users,
                     items=self.dataset.df_tracks)
        # TODO: we should verify the shape of predictions respects top_k=20
        rlist = myRecList(model=model, dataset=dataset)
        report_path = rlist()
        return report_path

    def evaluate(
            self,
            model,
            seed: int = None,
            upload: bool = True,
            limit: int = 0,
            custom_RecList: RecList = None,
            debug=True,
            # these are additional arguments for training the model, if you need
            # to pass additional stuff
            **kwargs 
    ):

        print("Generating data folds.")

        self._random_state = int(time.time()) if not seed else seed
        self.model = model

        self._events_hash = self.dataset._events_hash
        self._users_hash = self.dataset._users_hash
        self._tracks_hash = self.dataset._tracks_hash

        num_folds = self.dataset._test_set['fold'].max() + 1

        if num_folds != 4 or limit != 0:
            print("\nWARNING: default values are not used - upload is disabled")
            upload = False
        # if upload, check we have the necessary credentials
        if upload:
            assert self.email
            assert self.participant_id
            assert self.aws_access_key_id
            assert self.aws_secret_access_key
            assert self.bucket_name

        fold_results_path = []
        for fold in range(num_folds):
            train_df = self.dataset._get_train_set(fold=fold)
            if debug:
                print('\nPerforming Training for fold {}/{}...'.format(fold + 1, num_folds))
            self.model.train(train_df, **kwargs)
            if debug:
                print('Performing Evaluation for fold {}/{}...'.format(fold + 1, num_folds))
            results_path = self._test_model(self.model, fold, limit=limit, custom_RecList=custom_RecList)

            fold_results_path.append(results_path)

        raw_results = []
        fold_results = defaultdict(list)
        for fold, results_path in enumerate(fold_results_path):
            with open(os.path.join(results_path, 'results', 'report.json')) as f:
                result = json.load(f)
            # save reclist output
            raw_results.append(result)
            # tests which we care about
            tests = ['HIT_RATE']
            # extract test results
            for test_data in result['data']:
                if test_data['test_name'] in tests:
                    fold_results[test_data['test_name']].append(test_data['test_result'])

        # compute means
        # TODO: CI computation
        agg_results = {test: np.mean(res) for test, res in fold_results.items()}
        # build final output dict
        out_dict = {
            'reclist_reports': raw_results,
            'results': agg_results,
            'hash': hash(self)
        }
        # TODO: dump data somewhere better?
        if self.email:

            local_file = '{}_{}.json'.format(self.email.replace('@', '_'), int(time.time() * 10000))
            with open(local_file, 'w') as outfile:
                json.dump(out_dict, outfile, indent=2)
            print('SUBMISSION RESULTS SAVE TO {}'.format(local_file))

            if upload:
                upload_submission(local_file,
                                  aws_access_key_id=self.aws_access_key_id,
                                  aws_secret_access_key=self.aws_secret_access_key,
                                  participant_id=self.participant_id,
                                  bucket_name=self.bucket_name)

    def __hash__(self):
        hash_inputs = [
            self._num_folds,
            self._events_hash,
            self._users_hash,
            self._tracks_hash,
            inspect.getsource(self.evaluate).lstrip(' ').rstrip(' '),
            inspect.getsource(self._test_model).lstrip(' ').rstrip(' '),
            inspect.getsource(self.dataset._get_test_set).lstrip(' ').rstrip(' '),
            inspect.getsource(self.dataset._get_train_set).lstrip(' ').rstrip(' '),
            inspect.getsource(self.dataset._generate_folds).lstrip(' ').rstrip(' '),
            inspect.getsource(self.__init__).lstrip(' ').rstrip(' '),
            inspect.getsource(self.__hash__).lstrip(' ').rstrip(' '),
        ]
        hash_input = '_'.join([str(_) for _ in hash_inputs])
        return int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
