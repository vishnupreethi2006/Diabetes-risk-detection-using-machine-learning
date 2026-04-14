from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model/diabetes_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        # ✅ FIXED LINE
        final_input = np.array([features])

        prediction = model.predict(final_input)[0]

        if prediction == 1:
            result = "⚠️ High Risk of Diabetes"
        else:
            result = "✅ Low Risk of Diabetes"

    except Exception as e:
        print(e)   # 👈 ADD THIS FOR DEBUG
        result = "Error: Invalid Input"

    return render_template("index.html", prediction_text=result ,form_data=request.form)

if __name__ == "__main__":
    app.run(debug=True)