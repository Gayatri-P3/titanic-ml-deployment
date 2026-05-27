from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

from flasgger import Swagger

app = Flask(__name__)
# swagger = Swagger(app) /apidocs URL will open

#custom swagger route insted of inbuild route which was apidocs but it is url/swagger/
swagger = Swagger(app, config={
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/"
})# Swagger UI will be available at /swagger/

model = pickle.load(open("model.pkl", "rb"))

# HOME PAGE (loads UI)
@app.route("/")
def home():
    return render_template("index.html")


# MAIN FULL PREDICTION API
@app.route("/predict", methods=["POST"])
def predict():

    """
    Titanic Survival Prediction API
    ---
    tags:
      - Prediction API

    parameters:
      - name: body
        in: body
        required: true

        schema:
          properties:

            Pclass:
              type: integer
              example: 3

            age:
              type: number
              example: 22

            fare:
              type: number
              example: 7.25

            sibsp:
              type: integer
              example: 1

            parch:
              type: integer
              example: 0

            gender:
              type: string
              example: male

            embarked:
              type: string
              example: S

    responses:
      200:
        description: Titanic prediction response
    """

    data = request.get_json()

    pclass = int(data["Pclass"])
    age = float(data["age"])
    fare = float(data["fare"])
    sibsp = int(data["sibsp"])
    parch = int(data["parch"])

    gender = data["gender"]

    if gender == "male":
        gender = 1

    elif gender == "female":
        gender = 0

    else:
        gender = 2

    embarked_map = {
        "S": 0,
        "C": 1,
        "Q": 2
    }

    embarked = embarked_map[data["embarked"]]

    family_size = sibsp + parch

    features = np.array([[
        pclass,
        age,
        gender,
        sibsp,
        parch,
        fare,
        embarked,
        family_size
    ]])

    pred = model.predict(features)[0]

    # PROBABILITY
    probability = round(
        model.predict_proba(features)[0][1] * 100,
        2
    )

    result = "Survived ✅" if pred == 1 else "Not Survived ❌"

    return jsonify({
        "prediction": result,
        "probability": probability
    })




# AGE ANALYSIS API
@app.route("/age-analysis", methods=["POST"])
def age_analysis():

    """
    Age Analysis API
    ---
    tags:
      - Feature APIs

    parameters:
      - name: body
        in: body
        required: true

        schema:
          properties:
            age:
              type: number
              example: 25

    responses:
      200:
        description: Age prediction response
    """

    data = request.get_json()

    age = float(data["age"])

    # VALIDATION
    if age < 0 or age > 100:
        return jsonify({
            "status": "error",
            "message": "Age must be between 0 and 100"
        }), 400

    # Default values
    pclass = 3
    gender = 1
    sibsp = 0
    parch = 0
    fare = 7.25
    embarked = 0

    family_size = sibsp + parch

    features = np.array([[
        pclass,
        age,
        gender,
        sibsp,
        parch,
        fare,
        embarked,
        family_size
    ]])

    pred = model.predict(features)[0]

    probability = round(
        model.predict_proba(features)[0][1] * 100,
        2
    )

    result = "Survived ✅" if pred == 1 else "Not Survived ❌"

    return jsonify({
        "feature": "Age",
        "value": age,
        "prediction": result,
        "probability": probability
    })




# FARE ANALYSIS API
@app.route("/fare-analysis", methods=["POST"])
def fare_analysis():


    """
    Fare Analysis API
    ---
    tags:
      - Feature APIs

    parameters:
      - name: body
        in: body
        required: true

        schema:
          properties:

            fare:
              type: number
              example: 100

    responses:
      200:
        description: Fare prediction response
    """

    data = request.get_json()

    fare = float(data["fare"])


  # VALIDATION
    if fare < 0 or fare > 1000:
        return jsonify({
            "error": "Fare must be between 0 and 1000"
        })


    # Default values
    pclass = 3
    age = 25
    gender = 1
    sibsp = 0
    parch = 0
    embarked = 0

    family_size = sibsp + parch

    features = np.array([[
        pclass,
        age,
        gender,
        sibsp,
        parch,
        fare,
        embarked,
        family_size
    ]])

    pred = model.predict(features)[0]

    probability = round(
        model.predict_proba(features)[0][1] * 100,
        2
    )

    result = "Survived ✅" if pred == 1 else "Not Survived ❌"

    return jsonify({
        "feature": "Fare",
        "value": fare,
        "prediction": result,
        "probability": probability
    })




# GENDER ANALYSIS API
@app.route("/gender-analysis", methods=["POST"])
def gender_analysis():

    """
    Gender Analysis API
    ---
    tags:
      - Feature APIs

    parameters:
      - name: body
        in: body
        required: true

        schema:
          properties:
            gender:
              type: string
              example: male

    responses:
      200:
        description: Gender prediction response
    """

    data = request.get_json()

    gender_input = data["gender"]

    if gender_input == "male":
        gender = 1
    elif gender_input == "female":
        gender = 0
    else:
        gender = 2


 # VALIDATION
    if gender_input not in ["male", "female", "other"]:
        return jsonify({
            "error": "Gender must be male, female, or other"
        })

    # Default values
    pclass = 3
    age = 25
    sibsp = 0
    parch = 0
    fare = 7.25
    embarked = 0

    family_size = sibsp + parch

    features = np.array([[
        pclass,
        age,
        gender,
        sibsp,
        parch,
        fare,
        embarked,
        family_size
    ]])

    pred = model.predict(features)[0]

    probability = round(
        model.predict_proba(features)[0][1] * 100,
        2
    )

    result = "Survived ✅" if pred == 1 else "Not Survived ❌"

    return jsonify({
        "feature": "Gender",
        "value": gender_input,
        "prediction": result,
        "probability": probability
    })




# PCLAS ANALYSIS API
@app.route("/pclass-analysis", methods=["POST"])
def pclass_analysis():

    """
    Passenger Class Analysis API
    ---
    tags:
      - Feature APIs

    parameters:
      - name: body
        in: body
        required: true

        schema:
          properties:

            Pclass:
              type: integer
              example: 1

    responses:
      200:
        description: Passenger class prediction response
    """

    data = request.get_json()

    pclass = int(data["Pclass"])


      # VALIDATION
    if pclass not in [1, 2, 3]:
        return jsonify({
            "error": "Pclass must be 1, 2, or 3"
        })

    # Default values
    age = 25
    gender = 1
    sibsp = 0
    parch = 0
    fare = 7.25
    embarked = 0

    family_size = sibsp + parch

    features = np.array([[
        pclass,
        age,
        gender,
        sibsp,
        parch,
        fare,
        embarked,
        family_size
    ]])

    pred = model.predict(features)[0]

    probability = round(
        model.predict_proba(features)[0][1] * 100,
        2
    )

    result = "Survived ✅" if pred == 1 else "Not Survived ❌"

    return jsonify({
        "feature": "Passenger Class",
        "value": pclass,
        "prediction": result,
        "probability": probability
    })


# VALIDATION ENGINE API
# ALL INPUT VALIDATION API
@app.route("/validate-input", methods=["POST"])
def validate_input():

    """
    Input Validation API
    ---
    tags:
      - Validation API

    parameters:
      - name: body
        in: body
        required: true

        schema:
          properties:

            Pclass:
              type: integer
              example: 3

            age:
              type: number
              example: 22

            fare:
              type: number
              example: 7.25

            sibsp:
              type: integer
              example: 1

            parch:
              type: integer
              example: 0

            gender:
              type: string
              example: male

            embarked:
              type: string
              example: S

    responses:
      200:
        description: Validation response
    """

    data = request.get_json()

    errors = []

    age = float(data["age"])
    fare = float(data["fare"])
    pclass = int(data["Pclass"])
    sibsp = int(data["sibsp"])
    parch = int(data["parch"])

    gender = data["gender"]
    embarked = data["embarked"]

    # AGE
    if age < 0 or age > 100:
        errors.append("Age must be between 0 and 100")

    # FARE
    if fare < 0 or fare > 1000:
        errors.append("Fare must be between 0 and 1000")

    # PCLASS
    if pclass not in [1, 2, 3]:
        errors.append("Pclass must be 1, 2, or 3")

    # GENDER
    if gender not in ["male", "female", "other"]:
        errors.append("Gender must be male, female, or other")

    # EMBARKED
    if embarked not in ["S", "C", "Q"]:
        errors.append("Embarked must be S, C, or Q")

    # SIBSP
    if sibsp < 0 or sibsp > 10:
        errors.append("sibsp must be between 0 and 10")

    # PARCH
    if parch < 0 or parch > 10:
        errors.append("parch must be between 0 and 10")

    if errors:
        return jsonify({
            "validation": "FAILED ❌",
            "errors": errors
        })

    return jsonify({
        "validation": "PASSED ✅"
    })



#RUN APPLCATION
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)