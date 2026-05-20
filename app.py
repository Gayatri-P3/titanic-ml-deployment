from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

# LOAD MODEL FILES

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# SIMPLE HTML UI (INLINE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Titanic Survival Predictor</title>
</head>
<body>
    <h2>Titanic Survival Prediction</h2>

    <form action="/predict" method="POST">

        <input name="Pclass" type="number" placeholder="Pclass" required><br><br>

        <select name="Sex">
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select><br><br>

        <input name="Age" type="number" placeholder="Age" required><br><br>

        <input name="SibSp" type="number" placeholder="SibSp" required><br><br>

        <input name="Parch" type="number" placeholder="Parch" required><br><br>

        <input name="Fare" type="number" placeholder="Fare" required><br><br>

        <select name="Embarked">
            <option value="S">S</option>
            <option value="C">C</option>
            <option value="Q">Q</option>
        </select><br><br>

        <button type="submit">Predict</button>
    </form>

    <h3>{{ result }}</h3>
</body>
</html>
"""

# HOME PAGE (UI)
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

# HTML FORM PREDICTION

@app.route("/predict", methods=["POST"])
def predict_html():

    try:
        data = {
            "Pclass": int(request.form["Pclass"]),
            "Sex": request.form["Sex"],
            "Age": float(request.form["Age"]),
            "SibSp": int(request.form["SibSp"]),
            "Parch": int(request.form["Parch"]),
            "Fare": float(request.form["Fare"]),
            "Embarked": request.form["Embarked"]
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]

        result = "Survived" if prediction == 1 else "Not Survived"

        return render_template_string(HTML_TEMPLATE, result=result)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=str(e))

# REST API ENDPOINT

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["GET", "POST"])
def predict_html():

    if request.method == "GET":
        return render_template_string(HTML_TEMPLATE)

    try:
        data = {
            "Pclass": int(request.form["Pclass"]),
            "Sex": request.form["Sex"],
            "Age": float(request.form["Age"]),
            "SibSp": int(request.form["SibSp"]),
            "Parch": int(request.form["Parch"]),
            "Fare": float(request.form["Fare"]),
            "Embarked": request.form["Embarked"]
        }

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]

        result = "Survived" if prediction == 1 else "Not Survived"

        return render_template_string(HTML_TEMPLATE, result=result)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=str(e))

# RUN APP
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)