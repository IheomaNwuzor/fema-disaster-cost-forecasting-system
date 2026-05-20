import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================
# LOAD ML DATASET
# =========================

df = pd.read_csv(
    "data/processed/ml_ready_dataset.csv"
)

# =========================
# CREATE FEATURES (X)
# =========================

X = df.drop(
    columns=[
        "disasterNumber",
        "Total_Obligated_Amount"
    ]
)

# =========================
# CREATE TARGET (y)
# =========================

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
# TRAIN MODEL
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

# =========================
# PRINT RESULTS
# =========================

print("\nLINEAR REGRESSION RESULTS")

print(f"\nR² Score: {r2}")

print(f"\nMAE: {mae}")

print(f"\nRMSE: {rmse}")