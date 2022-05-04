from sklearn.linear_model import Lasso
from sklearn import metrics
import pandas as pd
import numpy as np
from data import comp_dataset, study_dataset

#######################
# Competition Dataset #
#######################

# Get Data
X_train, y_train, X_val, y_val = comp_dataset.train_data()
X_test, y_test = comp_dataset.test_data()

# Train Model
clf = Lasso(alpha=0.1)
clf.fit(X_train, y_train)

# Evaluate on validation data
y_pred = clf.predict(X_val)
y_pred = np.floor(y_pred)
print("Validation Evaluation")
print("RMSE: ", metrics.mean_squared_error(y_pred, y_val))
print("Accuracy: ", metrics.accuracy_score(y_pred, y_val))
print("Micro F1: ", metrics.f1_score(y_pred, y_val, average="micro"))

# Evaluate on testing data
y_pred = clf.predict(X_test)
y_pred = np.floor(y_pred)
print("Test Evaluation")
print("RMSE: ", metrics.mean_squared_error(y_pred, y_test))
print("Accuracy: ", metrics.accuracy_score(y_pred, y_test))
print("Micro F1: ", metrics.f1_score(y_pred, y_test, average="micro"))

##########################
# Original Study Dataset #
##########################

# Get Data
X_train, y_train, X_val, y_val = study_dataset.train_data()
X_test, y_test = study_dataset.test_data()

# Train Model
clf = Lasso(alpha=0.1)
clf.fit(X_train, y_train)

# Evaluate on validation data
y_pred = clf.predict(X_val)
y_pred = np.floor(y_pred)
print("Validation Evaluation")
print("RMSE: ", metrics.mean_squared_error(y_pred, y_val))
print("Accuracy: ", metrics.accuracy_score(y_pred, y_val))
print("Micro F1: ", metrics.f1_score(y_pred, y_val, average="micro"))

# Evaluate on testing data
y_pred = clf.predict(X_test)
y_pred = np.floor(y_pred)
print("Test Evaluation")
print("RMSE: ", metrics.mean_squared_error(y_pred, y_test))
print("Accuracy: ", metrics.accuracy_score(y_pred, y_test))
print("Micro F1: ", metrics.f1_score(y_pred, y_test, average="micro"))