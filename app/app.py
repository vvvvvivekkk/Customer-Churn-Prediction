from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("../model/churn_model.pkl")
columns = joblib.load("../model/columns.pkl")

@app.route("/")
def home():
    return "Churn Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Create empty input with all columns
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Fill only provided values
    for key in data:
        if key in input_df.columns:
            input_df.at[0, key] = data[key]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "churn": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)