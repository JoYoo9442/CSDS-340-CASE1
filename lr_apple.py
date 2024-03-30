# Logistic Regression for Apple Quality dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Load the dataset
df = pd.read_csv('./Data/train.csv')
# df = df.drop(columns=['Weight', 'Crunchiness', 'Acidity'])
data = df.to_numpy()
X_train = data[:, 0:-1]
y_train = data[:, -1]

# Load test data
df_test = pd.read_csv('./Data/test.csv')
data_test = df_test.to_numpy()
X_test = data_test[:, 0:-1]
y_test = data_test[:, -1]

# Standardize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Standardize the test data
X_test = scaler.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.2,
                                                    random_state=42)

# Create the model
model = LogisticRegression(penalty='l2', C=.05)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
