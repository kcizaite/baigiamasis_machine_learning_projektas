from flask import Flask
from flask import render_template
from flask import request
from random_forest_classifier import model
from plotinimas import plotinimas


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prevencinistyrimas", methods=["GET", "POST"])
def pacientas():
    if request.method == "POST":
        age = request.form["age"]
        sex = request.form["sex"]
        cp = request.form["cp"]
        trestbps = request.form["trestbps"]
        chol = request.form["chol"]
        fbs = request.form["fbs"]
        restecg = request.form["restecg"]
        thalach = request.form["thalach"]
        exang = request.form["exang"]
        oldpeak = request.form["oldpeak"]
        slope = request.form["slope"]
        ca = request.form["ca"]
        thal = request.form["thal"]

        modelis = ([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if sex == "0":
            p_duomenys = f"Vyras {age} m."
        else:
            p_duomenys = f"Moteris {age} m."
        # Random Forest modelio pritaikymas
        if model.predict(modelis) == 0:
            pacientas = "Pacientas NEPATENKA į padidėjusios rizikos grupę sirgti širdies ir kraujagyslių ligomis"
        else:
            pacientas = "Pacientas PATENKA į padidėjusios rizikos grupę sirgti širdies ir kraujagyslių ligomis"
        return render_template("patientform.html", pacientas=pacientas, p_duomenys=p_duomenys)
    return render_template("patientform.html", pacientas=False)


@app.route("/modeliai")
def analitika():
    return render_template("analytics.html", url=plotinimas())


if __name__ == "__main__":
    app.debug = True
    app.run()
