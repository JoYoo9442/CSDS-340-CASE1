# Ensembling Techniques
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Load the training data
df = pd.read_csv('./Data/train.csv')
data = df.to_numpy()
X_train = data[:, 0:-1]
y_train = data[:, -1]

# Load test data
df_test = pd.read_csv('./Data/test.csv')
data_test = df_test.to_numpy()
X_test = data_test[:, 0:-1]
y_test = data_test[:, -1]

# Standardize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Standardize the test data
X_test = scaler.transform(X_test)

model1 = DecisionTreeClassifier(criterion='entropy',
                                max_depth=11,
                                random_state=42)
model2 = SVC(C=19, gamma='scale', kernel='rbf',
             probability=True, random_state=42)

# Create the ensemble model
combined = VotingClassifier(estimators=[('dt', model1), ('svm', model2)],
                            voting='soft', weights=[1, 10])

# Train the models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
combined.fit(X_train, y_train)

# Predict the test data
y_pred = combined.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
