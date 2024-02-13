# Simple Linear Regression
# Importing the libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load the glass dataset
data = pd.read_csv("C:/glass.csv")

# Split the dataset into features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model by using the Gaussian Naive Bayes algorithm
model = GaussianNB()
model.fit(X_train, y_train)

# Predict the classes on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
print("Accuracy of Naives bayes is:", accuracy_score(y_test, y_pred))
print("\nClassification Report of Naives bayes is:\n", classification_report(y_test, y_pred))