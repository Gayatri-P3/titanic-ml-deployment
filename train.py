#trains SVM and saves model

import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("train.csv")

df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]].dropna()
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#train model
model = SVC()
model.fit(X_train, y_train)

#accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc * 100)

pickle.dump(model, open("model.pkl", "wb"))
print("model.pkl saved successfully!")


#save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully!")