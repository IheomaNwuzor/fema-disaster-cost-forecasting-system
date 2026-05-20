import pandas as pd

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/full_dataset.csv"
)

# =========================
# CONVERT DATES
# =========================

df["declarationDate"] = pd.to_datetime(
    df["declarationDate"]
)

df["incidentBeginDate"] = pd.to_datetime(
    df["incidentBeginDate"]
)

df["incidentEndDate"] = pd.to_datetime(
    df["incidentEndDate"]
)

# =========================
# EXTRACT DATE FEATURES
# =========================

df["declaration_year"] = (
    df["declarationDate"].dt.year
)

df["declaration_month"] = (
    df["declarationDate"].dt.month
)

# =========================
# DROP ORIGINAL DATE COLUMNS
# =========================

df = df.drop(
    columns=[
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate"
    ]
)

# =========================
# ONE-HOT ENCODING
# =========================

df = pd.get_dummies(
    df,
    columns=[
        "state",
        "incidentType",
        "declarationType"
    ],
    drop_first=True
)

# CONVERT TO INTEGERS
bool_columns = df.select_dtypes(include=["bool"]).columns

df[bool_columns] = df[bool_columns].astype(int)

# =========================
# SAVE FEATURED DATASET
# =========================

df.to_csv(
    "data/processed/ml_ready_dataset.csv",
    index=False
)

# =========================
# PRINT RESULTS
# =========================

print("\nML Ready Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())