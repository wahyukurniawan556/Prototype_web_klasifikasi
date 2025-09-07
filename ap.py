from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load model
nb_model = joblib.load("naive_bayes_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Ambil input user dari form
        features = [float(x) for x in request.form.values()]
        data = pd.DataFrame([features])

        # Prediksi
        pred_nb = nb_model.predict(data)[0]
        pred_dt = dt_model.predict(data)[0]

        return render_template(
            "index.html",
            prediction_nb=f"Hasil Naive Bayes: {pred_nb}",
            prediction_dt=f"Hasil Decision Tree: {pred_dt}"
        )

if __name__ == "__main__":
    app.run(debug=True)
