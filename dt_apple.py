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


# Load the dataset
df = pd.read_csv('./Data/train.csv')
data = df.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]

# Load test data from kaggle website
df_test = pd.read_csv('./Data/test.csv')
data_test = df_test.to_numpy()
X_apple = data_test[:, 0:-1]
y_apple = data_test[:, -1]

# Combine the training and test data
X_all = np.concatenate((X, X_apple), axis=0)
y_all = np.concatenate((y, y_apple), axis=0)

# Standardize the test data
scaler = StandardScaler()
scaler.fit(X)
X_apple = scaler.transform(X_apple)
# Standardize the data
X = scaler.transform(X)

# Plot a validation curve to tune the hyperparmeter max_depth
decision_tree = DecisionTreeClassifier()
display = ValidationCurveDisplay.from_estimator(decision_tree, X, y,
                                                param_name = 'max_depth',
                                                param_range = np.arange(1, 100, 10))
display.plot()
plt.show()

# Create the model
model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11, random_state = 1)

# Train the model
model.fit(X, y)

# Output the model
print(model.feature_importances_)

# Test the model
y_pred = model.predict(X_apple)
print(accuracy_score(y_apple, y_pred))



# Perform pca for dimensionality reduction to reduce overfitting of validation curve
pca = PCA(n_components=2)
pca_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11, random_state = 1)

# Dimensionality reduction
X_train_pca = pca.fit_transform(X_all)
X_test_pca = pca.transform(X_apple)

# fitting the logistic regression model on the reduced dataset
pca_model.fit(X_train_pca, y_all)

# Output the model
print(pca_model.feature_importances_)

# Test the model
y_pred = pca_model.predict(X_test_pca)
print(accuracy_score(y_apple, y_pred))



# Construct the decision tree
# Create the model
model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11, random_state = 1)
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
