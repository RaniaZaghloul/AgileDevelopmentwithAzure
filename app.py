"""Flask application for housing price prediction"""

import logging  # Standard library imports

from flask import Flask, request, jsonify  # Third-party imports
from flask.logging import create_logger

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

def scale(payload):
    """Scales the input payload using StandardScaler."""
    LOG.info("Scaling Payload: %s", payload)
    scaler = StandardScaler().fit(payload)
    scaled_adhoc_predict = scaler.transform(payload)
    return scaled_adhoc_predict

@app.route("/")
def home():
    """Returns a simple HTML home page."""
    return "<h3>Sklearn Prediction Home</h3>"

@app.route("/predict", methods=['POST'])
def predict():
    """Handles prediction requests and returns model predictions."""
    try:
        # Load pretrained model
        clf = joblib.load("./Housing_price_model/LinearRegression.joblib")
    except FileNotFoundError:
        LOG.error("Model file not found.")
        return jsonify({"error": "Model not loaded"}), 500
    except joblib.externals.loky.process_executor.BrokenProcessPool:
        LOG.error("Joblib process pool error.")
        return jsonify({"error": "Model execution error"}), 500
    except ValueError as e:
        LOG.error("Value error: %s", str(e))
        return jsonify({"error": "Invalid input data"}), 400

    json_payload = request.json
    LOG.info("JSON payload received: %s", json_payload)
    inference_payload = pd.DataFrame(json_payload)
    LOG.info("Inference payload DataFrame: %s", inference_payload)
    scaled_payload = scale(inference_payload)
    prediction = list(clf.predict(scaled_payload))
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
