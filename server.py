from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import hashlib

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    session_time = float(data["session_time"])
    pages_viewed = float(data["pages_viewed"])
    scroll_depth = float(data["scroll_depth"])
    idle_time = float(data["idle_time"])

    # Security layer: hash interaction data
    raw = f"{session_time}{pages_viewed}{scroll_depth}{idle_time}"
    data_hash = hashlib.sha256(raw.encode()).hexdigest()

    features = np.array([[session_time,pages_viewed,scroll_depth,idle_time]])

    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "churn_probability": float(probability),
        "hash": data_hash
    })

if __name__ == "__main__":
    app.run(debug=True)