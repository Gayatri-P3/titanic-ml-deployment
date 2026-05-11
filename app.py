from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "Titanic SVM Model is running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([{
        "Pclass": data["Pclass"],
        "Sex": 1 if data["Sex"] == "male" else 0,
        "Age": data["Age"],
        "SibSp": data["SibSp"],
        "Parch": data["Parch"],
        "Fare": data["Fare"]
    }])

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)

    result = "Survived" if prediction[0] == 1 else "Not Survived"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


    