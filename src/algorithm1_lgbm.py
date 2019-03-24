import os
import sys
import lightgbm
import numpy as np
import pandas as pd

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


def main():
    get_data()


if __name__ == "__main__":
    main()
