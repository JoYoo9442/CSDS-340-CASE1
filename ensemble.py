# Ensembling Techniques
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

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