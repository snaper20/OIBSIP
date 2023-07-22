from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
your_trained_model='C:/Users/user/Documents/GitHub/OIBSIP/CAR_PRICE/CAR_PRICE.ipynb'
joblib.dump(your_trained_model,"path_to_your_model.pkl")
model = joblib.load("path_to_your_model.pkl")  # Update with the correct path to your binary model file


@app.route("/", methods=["GET"])
def index():
    # Serve the index.html file for the front-end
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Make sure the JSON keys match the feature names your model expects
    features = [data["make"], data["model_year"], data["mileage"], ...]  # Add other features
    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
