### Import librairies
import pandas as pd
import os
import joblib
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

### Path
processed_data_path = os.path.join("data", "processed_data/")
normalized_data_path = os.path.join("data", "normalized_data/")
models_path = os.path.join("models/")
metrics_path = os.path.join("metrics/")
data_path = os.path.join("data/")

### Import data
X_train_scaled = pd.read_csv(normalized_data_path + 'X_train_scaled.csv')
X_test_scaled = pd.read_csv(normalized_data_path + 'X_test_scaled.csv')
y_train = pd.read_csv(processed_data_path + 'y_train.csv')
y_test = pd.read_csv(processed_data_path + 'y_test.csv')

model = joblib.load(models_path + "xgb_model.joblib")

### Make predictions
y_pred_train = model.predict(X_train_scaled.values)
y_pred_test  = model.predict(X_test_scaled.values)

### Metrics
train_metrics = {
        "r2": r2_score(y_train, y_pred_train),
        "mse": mean_squared_error(y_train, y_pred_train),
        "mae": mean_absolute_error(y_train, y_pred_train)
}

test_metrics = {
        "r2": r2_score(y_test, y_pred_test),
        "mse": mean_squared_error(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test)
}

### Save scores
scores = {
    "train": train_metrics,
    "test": test_metrics
}

with open(metrics_path + "scores.json", "w") as f:
    json.dump(scores, f, indent=2)

### Save predictions
y_test['y_pred'] = y_pred_test
y_test['residual'] = y_test['silica_concentrate'] - y_test['y_pred']
y_test.to_csv(data_path + "prediction.csv", index=False)