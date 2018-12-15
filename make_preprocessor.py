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
                      remainder='passthrough',
                      sparse=True):
    """Make preprocessing pipeline for numeric, binary, and categorical features
       Returns preprocessor and number of features after transform
       Standardize numeric values
       Impute missing numeric and binary values using specified strategy
       One-hot or ordinal encode categorical features
       By default, keep columns not in feature lists

       df: pandas dataframe - used to find all categories in categorical features

       numeric_features: list of numeric feature columns

       binary_features: list of binary feature columns

       categorical_features: list of categorical feature columns

       strategy: {'mean', 'median', 'most_frequent'} - imputation strategy

       cat_transform: {'ohe', 'ord'} - one-hot or ordinal encode categorical

       remainder: {'drop', 'passthrough'} - drop or keep columns not transformed

       sparse: {True, False} - one-hot encoder return sparse or dense
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
        'ohe': ('ohe', OneHotEncoder(categories=categories, sparse=sparse)),
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

    # Calculate number of features after transform
    flatten = lambda l: [x for subl in l for x in subl]
    n_numeric_features = len(numeric_features)
    n_binary_features = len(binary_features)
    n_categorical_features = {
        'ohe': len(flatten(categories)),
        'ord': len(categorical_features)
    }
    n_features = (n_numeric_features +
                  n_binary_features +
                  n_categorical_features['cat_transform'])

    return preprocessor, n_features
