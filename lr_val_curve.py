# Plotting Validation curve for logistic regression
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('./Data/train.csv')
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]
# Sequential Feature Selection
sbs = SequentialFeatureSelector(
        LogisticRegression(penalty='l1', solver='liblinear'),
        direction='backward')
sbs.fit(X, y)
X = sbs.transform(X)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Plot the validation curve
display = ValidationCurveDisplay.from_estimator(
        LogisticRegression(penalty='l2'),
        X, y,
        param_name='C',
        param_range=np.logspace(-3, 10, 100))
display.plot()

plt.show()
