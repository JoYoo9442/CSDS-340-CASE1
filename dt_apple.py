# Decision Tree model for Apple Quality dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.decomposition import PCA


# Load the training dataset
df = pd.read_csv('./Data/train.csv')
data = df.to_numpy()

# Split the data into features and labels
X_train = data[:, 0:-1]
y_train = data[:, -1]

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.2,
                                                    random_state=42)

# Plot a validation curve to tune the hyperparmeter max_depth
decision_tree = DecisionTreeClassifier()
display = ValidationCurveDisplay.from_estimator(decision_tree, X_train, y_train,
                                                param_name = 'max_depth',
                                                param_range = np.arange(1, 100, 10))
display.plot()
plt.show()

# Create the model
model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11, random_state = 1)

# Train the model
model.fit(X_train, y_train)

# Output the model
print(model.feature_importances_)

# Test the model
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
