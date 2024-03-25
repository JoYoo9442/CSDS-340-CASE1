# SVM model for apple quality dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('./Data/train.csv')
# df = df.drop(columns=['Weight', 'Crunchiness', 'Acidity'])
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Load test data from kaggle website
df_test = pd.read_csv('./Data/test1.csv')
data_test = df_test.to_numpy()
X_apple = data_test[:, 0:-1]
y_apple = data_test[:, -1]

# Combine the training and test data
X_all = np.concatenate((X, X_apple), axis=0)
y_all = np.concatenate((y, y_apple), axis=0)

# Standardize the test data
scaler = StandardScaler()
scaler.fit(X_all)
X_apple = scaler.transform(X_apple)

# Standardize the data
X = scaler.transform(X)

# Split the test data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Create the model
model = SVC(C=19, gamma='scale', kernel='rbf')

# Train the model
model.fit(X, y)

# Test the model
y_pred = model.predict(X_apple)

# Output the model
print(model.support_vectors_)
print(accuracy_score(y_apple, y_pred))
