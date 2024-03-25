# SVM parameter tuning using GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Load the dataset
df = pd.read_csv('./Data/train.csv')
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# GridSearchCV
params = {'C': np.arange(10, 20, 2),
          'gamma': [0.01, 0.1, 1, 'scale', 'auto']}
model = SVC()
grid_search = GridSearchCV(model,
                           param_grid=params,
                           cv=5,
                           scoring='accuracy')
grid_search.fit(X, y)
print(grid_search.best_params_)
