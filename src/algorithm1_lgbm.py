import os
import lightgbm
import numpy as np
import pandas as pd
from datetime import datetime

# Metrics for that competition is ROC AUC
from sklearn.metrics import roc_auc_score

# Function that automatically generates stratified k-fold split
from sklearn.model_selection import StratifiedKFold


def get_data():
    data_dir = os.path.join("..", "data")

    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Keep target from train set as y, and features from train set as x
    y = df_train['target']
    x = df_train.drop(['ID_code', 'target'], axis=1)

    return x, y, df_test


def train_algo(x, y, n_folds, **params):
    # This array is a prediction generated for cv-subset on each fold
    y_oof = np.zeros(shape=(x.shape[0],))

    print(params)

    #
    for i, (train_ind, valid_ind) in enumerate(
            StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=66372895).split(x, y)):
        # Get train and validation subsets for current fold
        x_tr, y_tr = x.iloc[train_ind], y.iloc[train_ind]
        x_cv, y_cv = x.iloc[valid_ind], y.iloc[valid_ind]

        # Convert pandas dataframes into lightgbm datasets
        train_dataset = lightgbm.Dataset(x_tr, label=y_tr)
        valid_dataset = lightgbm.Dataset(x_cv, label=y_cv)

        # This is actual model
        model = lightgbm.train(params,
                               train_dataset,
                               valid_sets=valid_dataset,
                               num_boost_round=100,
                               early_stopping_rounds=100)

        # Predict on train subset and validation subsets
        p_tr = model.predict(x_tr)
        p_cv = model.predict(x_cv)
        print(f"Fold {i+1}:: TRAIN SCORE: {roc_auc_score(y_tr, p_tr)}  CV SCORE: {roc_auc_score(y_cv, p_cv)}")

        # Set prediction for current fold to the total prediction array
        y_oof[valid_ind] = p_cv

        # Make a prediction on test data for current fold
        # p_te += model.predict(df_test.drop(['ID_code'], axis=1))
        model.save_model(os.path.join(save_dir, f"lgmb-{i+1}.txt"))

    print(f"PREDICTED SCORE: {roc_auc_score(y, y_oof)}")


def main_predict(n_folds):
    _, _, df_test = get_data()

    p_te = np.zeros(shape=(df_test.shape[0],))

    for i in range(n_folds):
        print(f"Loading model {i+1}")
        model = lightgbm.Booster(model_file=os.path.join(save_dir, f"lgmb-{i+1}.txt"))

        p_te += model.predict(df_test.drop(['ID_code'], axis=1))

    # Normalize to get an average prediction on test set
    p_te /= n_folds

    # Generate submission file
    submission = pd.read_csv(os.path.join("..", "data", "sample_submission.csv"))
    submission['target'] = p_te
    submission.to_csv(os.path.join("..", "submissions",
                                   f"lightgbm-{datetime.now().strftime('%m-%d-%Y--%H:%M:%S')}.csv"), index=False)


def main():
    df_x, df_y, df_test = get_data()

    print(df_x.head())

    # Parameters for LightXGB algorithm
    parameters = {
        'application': 'binary',
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 1
    }

    train_algo(df_x, df_y, n_folds=5, **parameters)


if __name__ == "__main__":
    save_dir = os.path.join("..", "santander", "models", "algorithm1")
    # save_dir = os.path.join(".")

    # main()
    main_predict(n_folds=5)
