### Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import os

### Path
rawdata_path = os.path.join("data", "raw_data/")
processed_data_path = os.path.join("data", "processed_data/")


### Import raw data
df = pd.read_csv(rawdata_path + 'raw.csv')
# Drop date colmun
df = df.drop(columns=['date'])

### Split data
X = df.drop(columns=['silica_concentrate'])
y = df['silica_concentrate']

X_train, X_test, y_train, y_test =  train_test_split(X, 
                                                     y, 
                                                     test_size=0.2, 
                                                     random_state=1,
                                                     )
os.makedirs(processed_data_path, exist_ok=True)
### Save splitted data
X_train.to_csv(processed_data_path + 'X_train.csv', index=False)
X_test.to_csv(processed_data_path + 'X_test.csv', index=False)
y_train.to_csv(processed_data_path + 'y_train.csv', index=False)
y_test.to_csv(processed_data_path + 'y_test.csv', index=False)