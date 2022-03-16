import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle


main_file_df = pd.read_csv(r"static/data/heart.csv")
X = main_file_df.drop("target", axis=1)
y = main_file_df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier(n_estimators=100)
train_model = model.fit(X_train, y_train)
# Nustatome pateiktų bandymo ir apsimokymo duomenų vidutinį tikslumą.
score_rfc = train_model.score(X_test, y_test)
train_model.score(X_test, y_test)
train_model.score(X_train, y_train)
# Naudojame funkciją predict() norėdami pamatyti prognozes, kurias funkcija atlieka su bandymo duomenimis.
train_model.predict(X_test)
y_pred = train_model.predict(X_test)
metrics.confusion_matrix(y_test, y_pred)

with open("model.pkl", "wb") as pickle_out:
    pickle.dump(model, pickle_out)

