from sklearn.linear_model import Lasso
from sklearn import metrics
import pandas as pd
import numpy as np
import dataset

# Get Dataset
X_train, y_train, X_val, y_val = dataset.get_data()

clf = Lasso(alpha=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
y_pred = np.floor(y_pred)

print("Evaluation")
print("RMSE: ", metrics.mean_squared_error(y_pred, y_val))
print("Accuracy: ", metrics.accuracy_score(y_pred, y_val))
print("Micro F1: ", metrics.f1_score(y_pred, y_val, average="micro"))
