# Plotting Validation curve for logistic regression
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Create a range of values for the parameter
param_name = 'C'
param_range = np.logspace(-8, 3, 10)
train_scores, test_scores = validation_curve(LogisticRegression(), X, y, param_name=param_name, param_range=param_range, cv=10, scoring='accuracy')

print(train_scores)
print(test_scores)
