from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

# Continuous features
CONTINUOUS_FEATURES = ['LotArea', 'YearBuilt', '1stFlrSF', 'GrLivArea']

# Categorical features
KITCHEN_QUALITY_COLUMN = 'KitchenQual'
CATEGORICAL_FEATURES = ['Neighborhood', 'HouseStyle', 'OverallQual', 'OverallCond', KITCHEN_QUALITY_COLUMN]
KITCHEN_QUALITY_DICT = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}

# ************Preprocessing*****************


def preprocess_data(X, fit=False):
    """
    This function performs the preprocessing of features extracted.

    Args:
        param1: X as a dataset
        param2: fit as checking if the model has been fitted or not

    Returns:
        returns the train dataset.
    """
    # Initialize the scalers and encoders
    scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)

    # Initialize the imputers
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    if (fit):
        X_train_processed = preprocess_train(X, scaler, one_hot_encoder, numeric_imputer, categorical_imputer)
        save_models(scaler, one_hot_encoder, numeric_imputer, categorical_imputer)
    else:
        scaler, one_hot_encoder, numeric_imputer, categorical_imputer = load_models()
        X_train_processed = preprocess_test(X, scaler, one_hot_encoder, numeric_imputer, categorical_imputer)

    return X_train_processed


def preprocess_train(X, scaler, one_hot_encoder, numeric_imputer, categorical_imputer):
    """
    This function trains the all the features extracted.

    Args:
        parameters: multiple categorical features data are loaded

    Returns:
        returns the train processed data.
    """
    X_train_continuous = X[CONTINUOUS_FEATURES]
    X_train_continuous = numeric_imputer.fit_transform(X_train_continuous)
    scaler.fit(X_train_continuous)
    X_train_continuous = scaler.transform(X_train_continuous)

    X_train_categorical = X[CATEGORICAL_FEATURES].copy()
    X_train_categorical[KITCHEN_QUALITY_COLUMN] = X_train_categorical[KITCHEN_QUALITY_COLUMN].map(KITCHEN_QUALITY_DICT)
    X_train_categorical = categorical_imputer.fit_transform(X_train_categorical)
    X_train_categorical_encoded = one_hot_encoder.fit_transform(X_train_categorical[:, :-1])
    X_train_kitchen_quality = X_train_categorical[:, -1].reshape(-1, 1)
    X_train_processed = np.hstack((X_train_continuous, X_train_categorical_encoded, X_train_kitchen_quality))
    return X_train_processed


def preprocess_test(X, scaler, one_hot_encoder, numeric_imputer, categorical_imputer):
    """
    This function transform the extracted features values.

    Args:
        parameters: Associated features with their corresponding data
    Returns:
        returns the train dataset.
    """
    X_continuous = X[CONTINUOUS_FEATURES]
    X_continuous = numeric_imputer.transform(X_continuous)
    X_continuous = scaler.transform(X_continuous)

    X_categorical = X[CATEGORICAL_FEATURES].copy()
    X_categorical[KITCHEN_QUALITY_COLUMN] = X_categorical[KITCHEN_QUALITY_COLUMN].map(KITCHEN_QUALITY_DICT)
    X_categorical = categorical_imputer.transform(X_categorical)
    X_categorical_encoded = one_hot_encoder.transform(X_categorical[:, :-1])
    X_kitchen_quality = X_categorical[:, -1].reshape(-1, 1)
    X_train_processed = np.hstack((X_continuous, X_categorical_encoded, X_kitchen_quality))
    return X_train_processed


def save_models(scaler, one_hot_encoder, numeric_imputer, categorical_imputer):
    """
    This function stores the models value in joblib format for the reuse purpose.

    Args:
        param1: X as a dataset
        param2: fit as checking if the model has been fitted or not

    Returns:
        No return, just save the values in local.
    """
    joblib.dump(scaler, "../models/scaler.joblib")
    joblib.dump(one_hot_encoder, "../models/one_hot_encoder.joblib")
    joblib.dump(numeric_imputer, "../models/numeric_imputer.joblib")
    joblib.dump(categorical_imputer, "../models/categorical_imputer.joblib")


def load_models():
    """
    This function loads the stored joblib files from local.

    Args:
        parameters: Associated features with their corresponding data

    Returns:
        returns all the joblib files stored in local for processing.
    """
    scaler = joblib.load("../models/scaler.joblib")
    one_hot_encoder = joblib.load("../models/one_hot_encoder.joblib")
    numeric_imputer = joblib.load("../models/numeric_imputer.joblib")
    categorical_imputer = joblib.load("../models/categorical_imputer.joblib")
    return scaler, one_hot_encoder, numeric_imputer, categorical_imputer
