import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the glass dataset
get_data = pd.read_csv("C:/glass.csv")

# Split the dataset into features (X) and target (y)
X = get_data.iloc[:, :-1]
y = get_data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model using the linear SVM algorithm
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict the classes on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
print("Accuracy of linear SVM is :", accuracy_score(y_test, y_pred))
print("\nClassification Report of linear SVM is:\n", classification_report(y_test, y_pred))