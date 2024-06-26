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
X_train = data[:, 0:-1]
y_train = data[:, -1]

# Standardize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Create the model
model = SVC(C=19, gamma='scale', kernel='rbf')

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Output the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
