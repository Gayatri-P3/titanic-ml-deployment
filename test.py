import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Sample input
sample = pd.DataFrame([{
    "Pclass": 1,
    "Sex": 1,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100
}])

# Scale input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print(prediction)