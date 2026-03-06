from flask import Flask, request, jsonify
import pickle
import numpy as np
import hashlib
from crypto_utils import decrypt_data

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/fog_predict", methods=["POST"])
def fog_predict():

    encrypted = request.json["data"]

    # decrypt incoming data
    decrypted = decrypt_data(encrypted)

    session_time, pages, scroll, idle = map(float, decrypted.split(","))

    raw = f"{session_time}{pages}{scroll}{idle}"
    data_hash = hashlib.sha256(raw.encode()).hexdigest()

    features = np.array([[session_time,pages,scroll,idle]])

    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "churn_probability": float(probability),
        "hash": data_hash
    })

if __name__ == "__main__":
    app.run(port=6000)