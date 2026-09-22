import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = ['initial_moisture', 'final_moisture', 'moisture_gain', 'thickness_micron', 'wvtr', 'otr', 'evidence_quality']
CATEGORICAL_FEATURES = ['packaging_material']

def get_preprocessor():
    """
    Creates and returns the scikit-learn ColumnTransformer for preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='error'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
        
    return preprocessor

if __name__ == '__main__':
    print("Preprocessing pipeline module. Import `get_preprocessor` to use.")
