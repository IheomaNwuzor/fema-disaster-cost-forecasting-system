import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =========================
# LOAD ML DATASET
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
# TRAIN XGBOOST MODEL
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
# FEATURE IMPORTANCE
# =========================

importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_
    }
)

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

# =========================
# SAVE FEATURE IMPORTANCE
# =========================

importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# =========================
# PRINT TOP FEATURES
# =========================

print("\nTop 20 Important Features:")
print(importance_df.head(20))

# =========================
# PLOT TOP 20 FEATURES
# =========================

top_20 = importance_df.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    top_20["feature"],
    top_20["importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Top 20 XGBoost Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "reports/xgboost_feature_importance.png"
)

plt.close()