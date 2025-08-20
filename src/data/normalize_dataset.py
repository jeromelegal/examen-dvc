### Import libraries
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

### Path
processed_data_path = os.path.join("data", "processed_data/")
normalized_data_path = os.path.join("data", "normalized_data/")

### Import data
X_train = pd.read_csv(processed_data_path + 'X_train.csv')
X_test = pd.read_csv(processed_data_path + 'X_test.csv')

### Instance
scaler = StandardScaler()

### Normalize
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Make dataframes
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

### Save scaled data
os.makedirs(normalized_data_path, exist_ok=True)
X_train_scaled_df.to_csv(normalized_data_path + 'X_train_scaled.csv', index=False)
X_test_scaled_df.to_csv(normalized_data_path + 'X_test_scaled.csv', index=False)

