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

# 'district_id', 'vdcmun_id', 'ward_id',
int_columns = ['count_families','count_floors_pre_eq', 'count_floors_post_eq', 'age_building',
       'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq']
categ_columns = ['legal_ownership_status','land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'condition_post_eq', 'technical_solution_proposed']
binary_columns = ['has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel',
       'has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry',
       'has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other', 'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other']

numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

categ_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(dtype=np.float32))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, int_columns),
        ('categorical', categ_transformer, categ_columns),
        ('passthrough', 'passthrough', binary_columns)
    ]
)

def get_int_columns():
    return int_columns

def get_categ_columns():
    return categ_columns

def get_binary_columns():
    return binary_columns

def train_data():
    """
    Returns the test - val split
    """

    features_df = pd.read_csv(os.path.join(current_dir,"TRAIN_STUDY.csv"))

    # select only the columns we want
    X = features_df[ int_columns + categ_columns + binary_columns]
    y = features_df['damage_grade'].to_frame()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Size information:")
    print("\t Number of Training Samples: {}".format(X_train.shape[0]))
    print("\t Number of Validation Samples: {}".format(X_val.shape[0]))

    # preprocess the data
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val = preprocessor.fit_transform(X_val).astype(np.float32)
    y_train = (y_train['damage_grade'].to_numpy())
    y_val = (y_val['damage_grade'].to_numpy())
    return(X_train, y_train, X_val, y_val)

def test_data():
    """
    Returns the test data, transformed to usable data through the pipeline
    """

    features_df = pd.read_csv(os.path.join(current_dir,"TEST_STUDY.csv"))
    X = features_df[ int_columns + categ_columns + binary_columns]
    y = features_df['damage_grade'].to_frame()

    print("\t Number of Test Samples: {}".format(X.shape[0]))

    # preprocess the data
    X = preprocessor.fit_transform(X).astype(np.float32)
    y = (y['damage_grade'].to_numpy())
    return X, y
