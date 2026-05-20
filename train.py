#trains SVM and saves model

import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)


# Load data
df = pd.read_csv(r"D:\titanic-ml-deployment\Flask_web_app_titanic_deployment\data\train.csv")

# select columns
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]].copy()

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# One-hot encoding
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

#features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]


#save column names
columns = X.columns.tolist()

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# feature scaling fits only train 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)    
X_test = scaler.transform(X_test)


#hyperparameter tuning
param_grid = {"C": [1, 5, 10],"gamma": [0.01, 0.1, "scale"],"kernel": ["rbf"]}

grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)

#train model
grid.fit(X_train, y_train)

#best model
model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

#predictions 
y_pred = model.predict(X_test)


#accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

#explaintion
tn, fp, fn, tp = conf_matrix.ravel()

print("\n========== Detailed Metrics ==========")
print(f"TP (Correct Survived): {tp}")
print(f"TN (Correct Not Survived): {tn}")
print(f"FP (Wrong Survived Prediction): {fp}")
print(f"FN (Missed Survivors): {fn}")

#important observation
print("\nImportant Observation:")
print("For Titanic survival prediction, False Negatives are sensitive.")
print("Because the model predicts passenger did NOT survive")
print("when the passenger actually survived.")

# classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#train final model on entire dataset
model.fit(scaler.fit_transform(X), y)


# Save files 
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(columns, open("columns.pkl", "wb"))

print("Files saved successfully!")
