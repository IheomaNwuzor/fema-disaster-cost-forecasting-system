import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/ml_ready_dataset.csv"
)

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop(
    columns=[
        "disasterNumber",
        "Total_Obligated_Amount"
    ]
)

y = np.log1p(
    df["Total_Obligated_Amount"]
)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN FINAL MODEL
# =========================

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

# =========================
# SAVE FEATURE COLUMNS
# =========================

joblib.dump(
    X.columns.tolist(),
    "models/model_features.pkl"
)

print("\nModel saved successfully.")

print("\nSaved files:")
print("models/xgboost_model.pkl")
print("models/model_features.pkl")