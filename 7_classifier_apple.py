# Group 7 - Classification of Apple Quality Dataset using SVM
# Madeleine Clore, Jonathan Yoo
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load the training dataset
df = pd.read_csv('./Data/train.csv')
data = df.to_numpy()

# Split the data into features and labels
X_train = data[:, 0:-1]
y_train = data[:, -1]

# Load test data
df_test = pd.read_csv('./Data/test.csv')
data_test = df_test.to_numpy()

# Split the test data into features and labels
X_test = data_test[:, 0:-1]
y_test = data_test[:, -1]

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Normalize the test data
X_test = scaler.transform(X_test)

# Create the model with selected hyperparameters
model = SVC(C=19, gamma='scale', kernel='rbf', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Output the model
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)*100}%")
