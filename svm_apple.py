# SVM model for apple quality dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Standardize the test data
scaler = StandardScaler()
X_apple = scaler.fit_transform(X_apple)

# Split the test data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_apple, y_apple,
                                                    test_size=0.2,
                                                    random_state=42)

# Create the model
model = SVC()

# Train the model
model.fit(X, y)

# Test the model
y_pred = model.predict(X_test)

# Output the model
print(model.support_vectors_)
print(accuracy_score(y_test, y_pred))
