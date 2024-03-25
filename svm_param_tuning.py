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

# Load test data from kaggle website
df_test = pd.read_csv('./Data/test.csv')
data_test = df_test.to_numpy()
X_test = data_test[:, 0:-1]
y_test = data_test[:, -1]

# Standardize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Standardize the test data
X_test = scaler.transform(X_test)

# Combine the training and test data
X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# Split the test data into training and testing set
if input("Randomize test data? (y/n): ") == 'y':
    X_not, X_test, y_not, y_test = train_test_split(X_all, y_all,
                                                    test_size=float(input(
                                                        "Enter test ratio: ")))

# Validation curve for C
param_range = np.logspace(-1, 5, 10)
display = ValidationCurveDisplay.from_estimator(
    SVC(), X_train, y_train, param_name="C",
    param_range=param_range
)

display.plot()
plt.show()
