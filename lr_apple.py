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
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Load test data from kaggle website
df_test = pd.read_csv('./Data/apple_quality.csv')
df_test.Quality = df_test.Quality.map({'good': 1, 'bad': 0})
data_test = df_test.to_numpy()
X_apple = data_test[:, 1:-1]
y_apple = data_test[:, -1]

# Normalize the data
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()
norm_scaler = MinMaxScaler()
X = robust_scaler.fit_transform(X)
X = standard_scaler.fit_transform(X)
X = norm_scaler.fit_transform(X)

# # Sequential Feature Selection
# sbs = SequentialFeatureSelector(
#         LogisticRegression(penalty='l1', solver='liblinear', C=0.05),
#         direction='forward')
# sbs.fit(X, y)
# X = sbs.transform(X)

# # Normalize the test data
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()
norm_scaler = MinMaxScaler()
X_apple = robust_scaler.fit_transform(X_apple)
X_apple = standard_scaler.fit_transform(X_apple)
X_apple = norm_scaler.fit_transform(X_apple)
# X_apple = sbs.transform(X_apple)

# Split the test data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_apple, y_apple,
                                                    test_size=0.8)

# Dont need to split the data into training and testing set

# Create the model
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.05)

# Train the model
model.fit(X, y)

# Output the model
print(model.coef_)

# Test the model
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
