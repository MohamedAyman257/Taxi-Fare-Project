
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DROP_COLS = [
    "user_id", "key", "pickup_datetime", "user_name", "driver_name",
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    "car_condition", "weather", "traffic_condition",
]
class clean_data(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None) -> "clean_data":
        return self  # stateless — nothing to learn
 
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
 
        # 1. Normalise column names
        X.columns = (
            X.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
        )
 
        # 2. Drop irrelevant columns
        X = X.drop(columns=DROP_COLS, errors="ignore")
 
        # 3. Weekend flag
        X["is_weekend"] = X["weekday"].isin([5,6]).astype(int)
        X = X.drop(columns=["weekday"], errors="ignore")
 
        # 4. Log-transform distance
        X["distance"] = np.log1p(X["distance"])
 
        return X
