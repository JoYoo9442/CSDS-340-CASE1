# Decision Tree model for Apple Quality dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('train.csv')
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dont need to split the data into training and testing set

# Create the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# Output the model
print(model.feature_importances_)
