import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

df = pd.read_csv("data/processed/ml_ready_dataset.csv")

X = df.drop(
    columns=[
        "disasterNumber",
        "Total_Obligated_Amount"
    ]
)

y = np.log1p(df["Total_Obligated_Amount"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

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

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nXGBOOST RESULTS")
print(f"\nR² Score: {r2}")
print(f"\nMAE: {mae}")
print(f"\nRMSE: {rmse}")