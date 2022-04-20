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


    # categorical data
    categ_features = ["position", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "plan_configuration", "legal_ownership_status", 'land_surface_condition']

    # get categorical data
    X1 = train_vals_df[categ_features].to_numpy()
    X_test1 = test_features_df[categ_features].to_numpy()
    enc = OrdinalEncoder()
    enc.fit(X1)
    X1 = enc.transform(X1)
    X_test1 = enc.transform(X_test1)
    
    numerical_features = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
    
    ## get numerical data
    df = train_vals_df[numerical_features]

    dftest = test_features_df[numerical_features]

    # normalize numerical data
    min_max_scaler = preprocessing.MinMaxScaler()
    X2 = min_max_scaler.fit_transform(df.values)
    X_test2 = min_max_scaler.transform(dftest.values)

    geo_id_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    X3 = train_vals_df[geo_id_features]
    X_test3 = test_features_df[geo_id_features]

    # binary features
    binary_columns = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                  'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
                  'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
                  'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_engineered',
                  'has_superstructure_other']
    X4 = train_vals_df[binary_columns]
    X_test4 = test_features_df[binary_columns]


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