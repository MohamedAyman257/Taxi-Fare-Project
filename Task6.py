import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from predictor.pipeline_cleandata import clean_data
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import joblib


DATA_PATH = os.path.join(os.getcwd(), '../Dataset','final_internship_data.csv')

DIST_COLS    = ["jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist"]
NUMERIC_COLS = ["passenger_count", "hour", "day", "month", "is_weekend",
                "year", "distance", "bearing"]

class TaxiFareTrainer:
    def __init__(
        self,
        data_path:    str   = DATA_PATH,
        Max_fare:     float = 200,
        Max_distance: float = 20,
        test_size:    float = 0.2,
        random_state: int   = 42,
    ):
        self.data_path    = data_path
        self.fare_cap     = Max_fare
        self.distance_cap = Max_distance
        self.test_size    = test_size
        self.random_state = random_state
 
    
        self.df      = None
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.pipe    = None
        self.metrics = None
 
    # ── Data ──────────────────────────────────
    def load_data(self):
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
        df.dropna(inplace=True)
        df = df[df["fare_amount"] >= 0]
        self.df = df
        return self
 
    def split_data(self):
        X = self.df.drop(columns=["fare_amount"])
        y = self.df["fare_amount"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return self
 
    def remove_outliers(self):
        mask = (
            (self.y_train <= self.fare_cap) &
            (self.X_train["distance"] <= self.distance_cap)
        )
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        return self
 
    # ── Pipeline ──────────────────────────────
    def _build_pipeline(self):
        preprocessor = ColumnTransformer(transformers=[
            ("airport_pca", Pipeline([
                ("scaler", RobustScaler()),
                ("pca",    PCA(n_components=1)),
            ]), DIST_COLS),
            ("numeric", RobustScaler(), NUMERIC_COLS),
        ])
 
        return Pipeline([
            ("clean", clean_data()),
            ("prep",  preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
            )),
        ])
 
    def train(self):
        print("Start Training.")
        self.pipe = self._build_pipeline()
        self.pipe.fit(self.X_train, np.log1p(self.y_train))
        print("Done Training.")
        return self
 
    # ── Evaluation ────────────────────────────
    def evaluate(self):
        preds = np.expm1(self.pipe.predict(self.X_test))
        self.metrics = {
            "MAE":  mean_absolute_error(self.y_test, preds),
            "MSE":  mean_squared_error(self.y_test, preds),
            "RMSE": root_mean_squared_error(self.y_test, preds),
            "R2":   r2_score(self.y_test, preds),
        }
        print("Metrics:", self.metrics)
        return self
 
    # ── save the pipeline ───────────────────────────
    def save(self):
        joblib.dump(self.pipe, "taxi_pipeline.pkl")
        return self
 
 
    def run(self):
        # Run the full training file #
        return (
            self.load_data()
                .split_data()
                .remove_outliers()
                .train()
                .evaluate()
                .save()
        )
 
 
 
TaxiFareTrainer().run()