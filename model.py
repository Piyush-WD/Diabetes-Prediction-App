import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset (make sure file name is correct)
diabetes_dataset = pd.read_csv("(Copy) diabetes.csv")

# Features and labels
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Save model and scaler
with open("diabetes.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")
