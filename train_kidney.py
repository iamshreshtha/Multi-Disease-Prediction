import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("kidney_disease.csv")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Drop ID column
df.drop("id", axis=1, inplace=True)

# Remove rows with missing target
df.dropna(subset=["classification"], inplace=True)

# Clean target labels
df["classification"] = (
    df["classification"]
    .astype(str)
    .str.strip()
    .map({"ckd": 1, "notckd": 0})
)

df.dropna(subset=["classification"], inplace=True)

# Features and target
X = df.drop("classification", axis=1)
y = df["classification"].astype(int)

# Convert numeric columns
numeric_cols = [
    'age', 'bp', 'sg', 'al', 'su',
    'bgr', 'bu', 'sc', 'sod', 'pot',
    'hemo', 'pcv', 'wc', 'rc'
]

for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save
joblib.dump(model, "kidney.pkl")

print("Kidney model saved.")