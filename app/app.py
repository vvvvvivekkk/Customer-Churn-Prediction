from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("../model/churn_model.pkl")

@app.route("/")
def home():
    return "Churn Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(list(data.values())).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "churn": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)