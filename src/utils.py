import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from datetime import datetime


def preprocess(df):
    '''

    :param df: pandas dataframe of events
    :return input_features: a dict of feature arrays to build matrices
    '''

    # clean up the timestamps and sort events
    df['timestamp'] = [datetime.fromtimestamp(ts / 1000) for ts in df['timestamp']]
    df = df.sort_values(['visitorid', 'timestamp'], ascending=True).reset_index(drop=True)

    # filter for users with more than 1 interaction
    interaction_counts = pd.DataFrame(df.groupby(['visitorid'])['event'].count()).reset_index()
    interaction_counts.columns = ['visitorid', 'count']
    visitorids_to_use = interaction_counts[interaction_counts['count']>1]['visitorid']
    df = df[df['visitorid'].isin(visitorids_to_use)]

    # create some arbitrary scores
    # this is just as a means to discriminate between interaction types (no science here)
    scores = {'view': 1, 'addtocart': 2, 'transaction': 3}
    df['event_score'] = df['event'].map(scores)

    # encoding item ids
    input_features = dict()
    input_features['event_score'] = np.array(df['event_score'])
    for col in ['visitorid', 'itemid']:
        encoder = LabelEncoder()
        input_features[col] = encoder.fit_transform(df[col])

    print("Preprocess complete.")
    return input_features


def build_matrices(input_features):
    '''

    :param input_features: a dict of feature arrays to build matrices
    :return matrices: dict of train and test sparse matrices shape(number or visitors, number of items)
    '''

    unique_visitors = len(np.unique(input_features['visitorid']))
    unique_items = len(np.unique(input_features['itemid']))

    test_cutoff = int(unique_visitors * 0.9)

    matrices = dict()
    matrices['train'] = coo_matrix((input_features['event_score'][0:test_cutoff],
                        (input_features['visitorid'][0:test_cutoff],input_features['itemid'][0:test_cutoff])),
                        shape=(unique_visitors, unique_items))
    matrices['test'] = coo_matrix((input_features['event_score'][test_cutoff + 1::],
                        (input_features['visitorid'][test_cutoff + 1::],input_features['itemid'][test_cutoff + 1::])),
                        shape=(unique_visitors, unique_items))
    print("Matrices built.")
    return matrices

