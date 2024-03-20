# Logistic Regression for Apple Quality dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('train.csv')
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Load test data from kaggle website
df_test = pd.read_csv('apple_quality.csv')
df_test.Quality = df_test.Quality.map({'good': 1, 'bad': 0})
data_test = df_test.to_numpy()
X_apple = data_test[:, 1:-1]
y_apple = data_test[:, -1]

# for i in range(len(y_apple)):
#     if y_apple[i] == 'good':
#         y_apple[i] = 1
#     else:
#         y_apple[i] = 0

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Standardize the test data
scaler = StandardScaler()
X_apple = scaler.fit_transform(X_apple)

# Split the test data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_apple, y_apple, test_size=0.2, random_state=42)

# Dont need to split the data into training and testing set

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Output the model
print(model.coef_)

# Test the model
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
