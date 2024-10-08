# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

from house_prices.preprocess import preprocess_data


def build_model(data: pd.DataFrame) -> dict[str, str]:
    """
      This function is used to build the model from the selected dataframe carried out form preprocessing.

      Args:
          param1: dataframe

      Returns:
          training performance value.
      """

    # Feature selection
    TARGETED = 'SalePrice'
    # Split the data
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    # Split features and target from the train set
    X_train = train_set.drop(columns=[TARGETED])
    y_train = train_set[TARGETED]

    # Preprocess training data
    X_train_processed = preprocess_data(X_train, fit=True)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # Save the model
    joblib.dump(model, "../models/model.joblib")

    def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
        rmsle = np.sqrt(mean_squared_error(y_test, y_pred))
        return round(rmsle, precision)

    # Make predictions and evaluate the model
    y_pred_train = model.predict(X_train_processed)

    rmsle_train = compute_rmsle(y_train, y_pred_train)

    return rmsle_train
