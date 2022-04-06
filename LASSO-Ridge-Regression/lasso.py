from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import dataset

# Get Dataset
X_train, y_train, X_val, y_val = dataset.get_data()

clf = Lasso(alpha=0.4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

print("Evaluation")
print("R^2 val: ", clf.score(X_val, y_val))
print("RMSE: ", mean_squared_error(y_pred, y_val))
