import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("data/diabetes.csv")

print("Dataset Loaded Successfully!\n")

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
cols = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]

for col in cols:
    data[col] = data[col].replace(0, data[col].mean())

print("Missing values handled!\n")

# -------------------------------
# 3. Split Data
# -------------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Data Split Done!\n")

# -------------------------------
# 4. Feature Scaling
# -------------------------------
# -------------------------------


# -------------------------------
# 5. Train Model (UPDATED)
# -------------------------------


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

print("Model Trained Successfully!\n")

# -------------------------------
# 6. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("🔥 Accuracy:", accuracy)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7. Save Model + Scaler
# -------------------------------
pickle.dump(model, open("model/diabetes_model.pkl", "wb"))


print("\nModel and Scaler Saved Successfully!")