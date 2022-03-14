import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


main_file_df = pd.read_csv(r"data/heart.csv")
X = main_file_df.drop("target", axis=1)
y = main_file_df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=1000)
train_model = model.fit(X_train, y_train)
# Nustatome pateiktų bandymo ir apsimokymo duomenų vidutinį tikslumą.
score_lor = train_model.score(X_test, y_test)
train_model.score(X_test, y_test)
train_model.score(X_train, y_train)
y_pred = train_model.predict(X_test)
metrics.confusion_matrix(y_test, y_pred)
