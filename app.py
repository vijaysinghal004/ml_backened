import os
from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load the preprocessing components and model from the .pkl file
with open("preprocessing_and_model.pkl", "rb") as f:
    data = pickle.load(f)

# Extract individual components
le_soil = data["label_encoder_soil"]
le_month = data["label_encoder_month"]
le_crop = data["label_encoder_crop"]
scaler = data["scaler"]
model = data["model"]

@app.route("/")
def home():
    return "Welcome to the Flask app for crop recommendation!"

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input JSON data from the request
    data = request.get_json()

    # Extract input data
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    soil_type = data.get("soil_type")
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    month = data.get("month")

    # Check for missing or invalid inputs
    if None in [latitude, longitude, soil_type, temperature, humidity, month]:
        return jsonify({"error": "Missing input data"}), 400

    try:
        # Encode soil type and month using the label encoders
        soil_type_encoded = le_soil.transform([soil_type])[0]
        month_encoded = le_month.transform([month])[0]
    except ValueError:
        return jsonify({"error": "Invalid soil_type or month"}), 400

    # Create input array
    input_data = np.array([[latitude, longitude, soil_type_encoded, temperature, humidity, month_encoded]])

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    predictions = model.predict(input_data_scaled)
    predicted_class_index = np.argmax(predictions)

    # Decode the predicted class index back to crop name
    predicted_crop_name = le_crop.inverse_transform([predicted_class_index])[0]

    # Return the prediction as a JSON response
    return jsonify({"predicted_crop": predicted_crop_name})

if __name__ == "__main__":
    app.run()
