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
    test_size = input("Enter test ratio: ")
    if test_size == '':
        test_size = 0.2
    else:
        test_size = float(test_size)
    X_not, X_test, y_not, y_test = train_test_split(X_all, y_all,
                                                    test_size=test_size)

# Create the model
model = LogisticRegression(penalty='l2', C=0.05)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
