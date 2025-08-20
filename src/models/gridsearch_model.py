### Import libraries
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
import os
import pickle

### Path
processed_data_path = os.path.join("..", "..", "data", "processed_data/")

### Import data
X_train_scaled = pd.read_csv(processed_data_path + 'X_train_scaled.csv')
y_train = pd.read_csv(processed_data_path + 'y_train.csv')

### Base model instance
# https://xgboost.readthedocs.io/en/stable/parameter.html#global-configuration
xgb = XGBRegressor(
    random_state=42,
    n_jobs=-1
)

### Parameters grid
params_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

### GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=params_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)

### Train
grid.fit(X_train_scaled.values, y_train.values)

### Save best params
with open("../models/best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)

### Save CV results
with open("cv_results.pkl", "wb") as f:
    pickle.dump(grid.cv_results_, f)

cv_df = pd.DataFrame(grid.cv_results_)
cv_df.to_csv("cv_results.csv", index=False)

