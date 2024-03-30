# SVM parameter tuning using GridSearchCV
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

# Standardize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Validation curve for C
param_range = np.logspace(1, 2, 10)
display = ValidationCurveDisplay.from_estimator(
    SVC(), X_train, y_train, param_name="C",
    param_range=param_range
)

display.plot()
plt.show()
