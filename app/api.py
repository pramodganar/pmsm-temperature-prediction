
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "PMSM Temperature Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({"prediction": 42})

if __name__ == "__main__":
    app.run(debug=True)
