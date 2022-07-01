import numpy as np
from pyexpat import model
from flask import Flask, render_template, jsonify, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template ("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    return render_template ("index.html", prediction_text = "Sales : {}".format(output))

if __name__=="__main__":
    app.run(debug=True)