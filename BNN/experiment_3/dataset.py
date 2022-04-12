import pandas as pd
import numpy as np
import os
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def get_data():

    data_dir = "../../data/"
    train_vals_df = pd.read_csv(os.path.join(data_dir, "train_values.csv"))
    train_labels_df = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
    test_features_df = pd.read_csv(os.path.join(data_dir, "test_values.csv"))
    test_building_ids = test_features_df[["building_id"]]

    # get categorical data
    X1 = train_vals_df[["position", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "plan_configuration", "legal_ownership_status"]].to_numpy()
    X_test1 = test_features_df[["position", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "plan_configuration", "legal_ownership_status"]].to_numpy()
    enc = OrdinalEncoder()
    enc.fit(X1)
    X1 = enc.transform(X1)
    X_test1 = enc.transform(X_test1)
    

    ## get numerical data
    df = train_vals_df[[
    'count_floors_pre_eq',
    'age',
    'area_percentage',
    'height_percentage']]

    dftest = test_features_df[[
    'count_floors_pre_eq',
    'age',
    'area_percentage',
    'height_percentage']]

    # normalize numerical data
    min_max_scaler = preprocessing.MinMaxScaler()
    X2 = min_max_scaler.fit_transform(df.values)
    X_test2 = min_max_scaler.transform(dftest.values)

    X3 = train_vals_df[['geo_level_1_id', 'geo_level_2_id']]
    X_test3 = test_features_df[['geo_level_1_id', 'geo_level_2_id']]


    # get labels

    y = train_labels_df["damage_grade"].to_numpy() - 1
    X = np.concatenate((X3,X1,X2,), axis = 1)
    X_test = np.concatenate((X_test3, X_test1, X_test2), axis=1)

    rng = np.random.RandomState(7)
    perm = rng.permutation(range(len(X)))
    X_rand = X[perm]
    y_rand = y[perm]

    trainIdx = int(.7*len(X))
    valIdx = int(.85*len(X))
    X_train = X[:trainIdx]
    y_train = y[:trainIdx]
    X_val = X[trainIdx:valIdx]
    y_val = y[trainIdx:valIdx]
    return(X_train, y_train, X_val, y_val, X_test, test_building_ids)

if __name__ == "__main__":

    get_data()