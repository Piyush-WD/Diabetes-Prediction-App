import numpy as np
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("diabetes.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("dp_html.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    float_features = [float(x) for x in request.form.values()]
    
    # Scale input
    features = scaler.transform([float_features])

    # Predict
    prediction = model.predict(features)[0]

    # Output text
    output = "non-diabetic" if prediction == 0 else "diabetic"

    return render_template("dp_html.html", prediction_text=f"The person is {output}")


if __name__ == "__main__":
    flask_app.run(debug=True)
