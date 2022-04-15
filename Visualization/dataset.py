"""
Author: Pierce Howell
Modified: Randall Kliman 4/6/2022 
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(__file__)
features_df = pd.read_csv(os.path.join(current_dir, "../data/train_values.csv"))
labels_df = pd.read_csv(os.path.join(current_dir, "../data/train_labels.csv"))


int_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
               'count_floors_pre_eq', 'age', 'area_percentage', 
               'height_percentage']
categ_columns = ['land_surface_condition', 'foundation_type', 'roof_type',
                 'ground_floor_type', 'other_floor_type', 'position',
                 'plan_configuration',
                 ]
binary_columns = ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                  'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
                  'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
                  'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_engineered',
                  'has_superstructure_other'
                  ]

def get_data():
    """
    Returns the test - val split
    """

    features_df = pd.read_csv(os.path.join(current_dir, "../data/train_values.csv"))
    labels_df = pd.read_csv(os.path.join(current_dir, "../data/train_labels.csv"))

    # select only the columns we want
    X = features_df[ int_columns + categ_columns + binary_columns]
    y = labels_df['damage_grade'].to_frame()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Size information:")
    print("\t Number of Training Samples: {}".format(X_train.size))
    print("\t Number of Validation Samples: {}".format(X_val.size))

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categ_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(dtype=np.float32))
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, int_columns),
            # ('categorical', categ_transformer, categ_columns),
            # ('passthrough', 'passthrough', binary_columns)
        ]
    )

    # preprocess the data
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val = preprocessor.fit_transform(X_val).astype(np.float32)
    y_train = (y_train['damage_grade'].to_numpy())
    y_val = (y_val['damage_grade'].to_numpy())
    return(X_train, y_train, X_val, y_val)