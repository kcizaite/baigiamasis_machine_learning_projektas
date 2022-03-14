from kneighbors_classifier import *
from logistic_regression import *
from linear_regression import *
from random_forest_classifier import *
import matplotlib.pyplot as plt
import base64
import io


def plotinimas():
    names = ["Linear Regression", "Random Forest Classifier", "KNeighbors Classifier", "Logistic Regression"]
    values = [score_lir, score_rfc, score_knc, score_lor]

    plt.rcParams.update({"figure.autolayout": True})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, values, color=["lightskyblue", "limegreen", "darkorange", "indianred"])
    ax.set(xlim=[0, 1], xlabel="Performance", ylabel="Models")
    for index, value in enumerate(values):
        plt.text(value, index, round(value, 2))

    plt.rc("font", size=10)
    plt.rc("axes", titlesize=10)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    # Paveiksliuko konfiguracija
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")

    return plot_url
