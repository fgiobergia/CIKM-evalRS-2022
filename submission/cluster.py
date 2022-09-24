import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def get_users_clusters(train, df_users, n_clusters=2):
    df = df_users.loc[list(set(train["user_id"]))]
    user_ids = df.index


    discard = ['registered_unixtime', ]

    countries = pd.read_csv("countries.txt", keep_default_na=False)
    countries = pd.concat([countries, pd.DataFrame(columns=countries.columns, data=[["n/a"]*len(countries.columns)])])
    df = df.fillna({"country":"n/a"}).merge(countries, left_on="country", right_on="Two_Letter_Country_Code")


    cols = ['age', 'gender', 'playcount', "Continent_Code",
        'novelty_artist_avg_month',
        'novelty_artist_avg_6months', 'novelty_artist_avg_year',
        'mainstreaminess_avg_month', 'mainstreaminess_avg_6months',
        'mainstreaminess_avg_year', 'mainstreaminess_global',
        'cnt_listeningevents', 'cnt_distinct_tracks', 'cnt_distinct_artists',
        'cnt_listeningevents_per_week', 'relative_le_per_weekday1',
        'relative_le_per_weekday2', 'relative_le_per_weekday3',
        'relative_le_per_weekday4', 'relative_le_per_weekday5',
        'relative_le_per_weekday6', 'relative_le_per_weekday7',
        'relative_le_per_hour0', 'relative_le_per_hour1',
        'relative_le_per_hour2', 'relative_le_per_hour3',
        'relative_le_per_hour4', 'relative_le_per_hour5',
        'relative_le_per_hour6', 'relative_le_per_hour7',
        'relative_le_per_hour8', 'relative_le_per_hour9',
        'relative_le_per_hour10', 'relative_le_per_hour11',
        'relative_le_per_hour12', 'relative_le_per_hour13',
        'relative_le_per_hour14', 'relative_le_per_hour15',
        'relative_le_per_hour16', 'relative_le_per_hour17',
        'relative_le_per_hour18', 'relative_le_per_hour19',
        'relative_le_per_hour20', 'relative_le_per_hour21',
        'relative_le_per_hour22', 'relative_le_per_hour23']

    df = df[cols]

    # fix nans
    df.loc[df["age"]==-1,"age"] = df[df["age"]!=-1]["age"].mean()

    df.fillna({"Continent_Code": "n/a", "gender": "n/a"}, inplace=True)
    df.fillna(0, inplace=True)# is this ok?


    df["playcount"] += 2
    logged = [
        "playcount",
        'cnt_listeningevents', 'cnt_distinct_tracks', 'cnt_distinct_artists',
        'cnt_listeningevents_per_week'
    ]
    df.loc[:, logged] = np.log(df[logged])

    df_float = df.select_dtypes(float)
    df_float = (df_float - df_float.min()) / (df_float.max() - df_float.min())

    df_obj = df.select_dtypes(exclude=float)

    df_final = pd.concat([df_float, pd.get_dummies(df_obj)], axis=1)

    km = KMeans(n_clusters)
    labels = km.fit_predict(df_final)

    return pd.Series(index=user_ids, data=labels, name="cluster")