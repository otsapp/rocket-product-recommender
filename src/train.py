import argparse
import os
from datetime import date
from sklearn.externals import joblib
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k

from utils import preprocess, build_matrices

def model_fn(model_dir):
    '''
    Function to load in models, useful for AWS deployment.
    :param model_dir: models directory
    :return: models
    '''
    print("Loading models.")

    # load using joblib
    model = joblib.load(os.path.join(model_dir, "models.joblib"))
    print("Done loading models.")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, help='directory of models', default='models')
    parser.add_argument('--data-dir', type=str, help='directory of data', default='../data/')

    # models related args
    parser.add_argument('--no-components', type=int, default=5)
    parser.add_argument('--loss-method', type=str, default='warp')
    parser.add_argument('--learning-schedule', type=str, default='adagrad')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_threads', type=int, default=1)

    args = parser.parse_args()

    # date of training
    current_date = date.today()

    # load dataset
    df = pd.read_csv(os.path.join(args.data_dir, 'events.csv'))

    # building preprocessed dataframe
    df_processed = preprocess(df)

    # Save the item ids used in training as a dataframe to be used for predicting new users later
    #item_ids = df_processed.loc[:, ['itemid_enc', 'itemid']].drop_duplicates()
    #joblib.dump(item_ids, os.path.join(args.data_dir, "item_id_training_data.joblib"))

    # build user-item matrices for training and testing
    matrices = build_matrices(df_processed)

    # define models
    model = LightFM(no_components=args.no_components, loss=args.loss_method, learning_schedule=args.learning_schedule)

    # fit models
    model.fit(matrices['train'], epochs=100, num_threads=1)

    # Save the trained models
    os.mkdir(args.model_dir)
    joblib.dump(model, os.path.join(args.model_dir, f"model_{current_date}.joblib"))
    print(f"Model save successful, path: models/model_{current_date}.joblib")

    # evaluation metrics
    # This precision metric is prefered as an eval metric for 'warp' loss method
    # However the score is far to low for this particular model, so reverting to AUC

    if args.loss_method == None:
        # the warp loss function optimises for precision at k.
        train_precision = precision_at_k(model, matrices['train'], k=5).mean()
        test_precision = precision_at_k(model, matrices['test'], k=5).mean()
        print("Train precision: %.2f" % train_precision)
        print("Test precision: %.2f" % test_precision)
    else:
        auc_train = auc_score(model, matrices['train']).mean()
        auc_test = auc_score(model, matrices['test']).mean()
        print("Train AUC score: %.2f" % auc_train)
        print("Test AUC score: %.2f" % auc_test)