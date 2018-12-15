import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def make_preprocessor(df,
                      numeric_features,
                      binary_features,
                      categorical_features,
                      strategy='median',
                      cat_transform='ohe',
                      remainder='passthrough'):
    """Make preprocessing pipeline for numeric, binary, and categorical features
       Standardize numeric values
       Impute missing numeric and binary values using specified strategy
       One-hot or ordinal encode categorical features
       By default, keep columns not in feature lists

       df: pandas dataframe - used to find all categories in categorical features

       numeric_features: list of numeric feature columns

       binary_features: list of binary feature columns

       categorical_features: list of categorical feature columns

       strategy: {'mean', 'median', 'most_frequent'} - imputation strategy

       cat_transform: {'ohe', 'ord'} - One-hot or ordinal encode categorical

       remainder: {'drop', 'passthrough'} - drop or keep columns not transformed
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=strategy)),
        ('scaler', StandardScaler())
    ])

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=strategy))
    ])

    categories = [df[feature].unique() for feature in categorical_features]

    cat_transformers = {
        'ohe': ('ohe', OneHotEncoder(categories=categories, sparse=True)),
        'ord': ('ord', OrdinalEncoder(categories=categories))
    }

    categorical_transformer = Pipeline(steps=[
        cat_transformers[cat_transform]
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', binary_transformer, binary_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder=remainder)

    return preprocessor
