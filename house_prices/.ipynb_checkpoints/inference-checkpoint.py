import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import preprocess_data


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
      This function is used to predict the value(SalePrice).

      Args:
          param1: input_data as a test datasets

      Returns:
          returns the predicted value.
      """
    # Load the model and preprocessors
    model = joblib.load("../models/model.joblib")

    # Preprocess inference data
    X_inference_processed = preprocess_data(input_data, fit=False)

    # Make predictions
    predictions = model.predict(X_inference_processed)
    return predictions
