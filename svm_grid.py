# Grid Search for SVM hyperparameter tuning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import ValidationCurveDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./Data/train.csv')
# df = df.drop(columns=['Weight', 'Crunchiness', 'Acidity'])
data = df.to_numpy()
X_train = data[:, 0:-1]
y_train = data[:, -1]

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 'scale', 'auto']
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid,
                    cv=5, scoring='accuracy')

grid.fit(X_train, y_train)

print("Best parameters: ", grid.best_params_)
