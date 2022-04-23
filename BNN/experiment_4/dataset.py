import pandas as pd
import numpy as np
import os
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# def get_data():

#     data_dir = "../../data/"
#     train_vals_df = pd.read_csv(os.path.join(data_dir, "train_values.csv"))
#     train_labels_df = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
#     test_features_df = pd.read_csv(os.path.join(data_dir, "test_values.csv"))
#     test_building_ids = test_features_df[["building_id"]]


#     # categorical data
#     categ_features = ["position", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "plan_configuration", "legal_ownership_status", 'land_surface_condition']

#     # get categorical data
#     X1 = train_vals_df[categ_features].to_numpy()
#     X_test1 = test_features_df[categ_features].to_numpy()
#     enc = OrdinalEncoder()
#     enc.fit(X1)
#     X1 = enc.transform(X1)
#     X_test1 = enc.transform(X_test1)
    
#     numerical_features = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
    
#     ## get numerical data
#     df = train_vals_df[numerical_features]

#     dftest = test_features_df[numerical_features]

#     # normalize numerical data
#     min_max_scaler = preprocessing.MinMaxScaler()
#     X2 = min_max_scaler.fit_transform(df.values)
#     X_test2 = min_max_scaler.transform(dftest.values)

#     geo_id_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
#     X3 = train_vals_df[geo_id_features]
#     X_test3 = test_features_df[geo_id_features]

#     # binary features
#     binary_columns = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
#                   'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
#                   'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
#                   'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_engineered',
#                   'has_superstructure_other']
#     X4 = train_vals_df[binary_columns]
#     X_test4 = test_features_df[binary_columns]


#     # get labels

#     y = train_labels_df["damage_grade"].to_numpy() - 1
#     X = np.concatenate((X3,X1,X2,X4), axis = 1)
#     X_test = np.concatenate((X_test3, X_test1, X_test2, X_test4), axis=1)

#     rng = np.random.RandomState(7)
#     perm = rng.permutation(range(len(X)))
#     X_rand = X[perm]
#     y_rand = y[perm]

#     trainIdx = int(.7*len(X))
#     valIdx = int(.85*len(X))
#     X_train = X[:trainIdx]
#     y_train = y[:trainIdx]
#     X_val = X[trainIdx:valIdx]
#     y_val = y[trainIdx:valIdx]
#     return(X_train, y_train, X_val, y_val, X_test, test_building_ids)

def get_data():
    data_dir = "../../data/"
    train_df = pd.read_csv(os.path.join(data_dir, "TRAIN.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "TEST.csv"))

    # features (ALL)
    categ_features = ["position", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "plan_configuration", "legal_ownership_status", 'land_surface_condition']
    numerical_features = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
    geo_id_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    binary_features = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
            'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
            'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_engineered',
            'has_superstructure_other']

    # transform categorical feautres
    X_train_categ = train_df[categ_features].to_numpy()
    X_test_categ = test_df[categ_features].to_numpy()
    enc = OrdinalEncoder()
    enc.fit(X_train_categ)
    X_train_categ = enc.transform(X_train_categ)
    X_test_categ = enc.transform(X_test_categ)

    # normalize numerical features
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_numer = train_df[numerical_features].values
    X_test_numer = test_df[numerical_features].values
    X_train_numer = min_max_scaler.fit_transform(X_train_numer)
    X_test_numer = min_max_scaler.transform(X_test_numer)

    # Get geoid and binary features
    X_train_geo = train_df[geo_id_features].values
    X_test_geo = test_df[geo_id_features].values
    X_train_bin = train_df[binary_features].values
    X_test_bin = test_df[binary_features].values

    X_train = np.concatenate((X_train_geo, X_train_categ, X_train_numer, X_train_bin), axis=1)
    #X_train = X_train[:100000]
    X_test = np.concatenate((X_test_geo, X_test_categ, X_test_numer, X_test_bin), axis=1)

    # get the labels
    y_train = train_df["damage_grade"].to_numpy() - 1
    #y_train = y_train[:100000]
    y_test = test_df["damage_grade"].to_numpy() - 1

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
    return X_train, y_train, X_val, y_val, X_test, y_test

class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            label = self.y[idx]
            x = self.X[idx]
            return x, label

        def delete(self, ind):

            self.X = np.delete(self.X, ind, 0)
            self.y = np.delete(self.y, ind, 0)

        def add(self, X, y):
            self.X = np.append(self.X, X, axis=0)
            self.y = np.append(self.y, y, axis=0)

if __name__ == "__main__":

    get_data()