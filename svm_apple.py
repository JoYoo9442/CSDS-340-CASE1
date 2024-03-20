# SVM model for apple quality dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('train.csv')
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Load test data from kaggle website
df_test = pd.read_csv('apple_quality.csv')
data_test = df_test.to_numpy()
X_test = data_test[:, 1:-1]
y_test = data_test[:, -1]

for i in range(len(y_test)):
    if y_test[i] == 'good':
        y_test[i] = 1
    else:
        y_test[i] = 0

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Standardize the test data
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Dont need to split the data into training and testing set

# Create the model
model = SVC()

# Train the model
model.fit(X, y)

# Test the model
y_pred = model.predict(X_test)

# Output the model
print(model.support_vectors_)
print(accuracy_score(y_test, y_pred))
