### Import libraries
import pandas as pd
from xgboost import XGBRegressor
import os
import pickle
import joblib

### Path
normalized_data_path = os.path.join("data", "normalized_data/")
processed_data_path = os.path.join("data", "processed_data/")
models_path = os.path.join("models/")

### Import data
X_train_scaled = pd.read_csv(normalized_data_path + 'X_train_scaled.csv')
y_train = pd.read_csv(processed_data_path + 'y_train.csv')

with open(models_path + "best_params.pkl", "rb") as f:
    params = pickle.load(f)

### Model
xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    **params
)    

### Train
xgb.fit(X_train_scaled.values, y_train.values)

### Save trained model
joblib.dump(xgb, models_path + "xgb_model.joblib")
