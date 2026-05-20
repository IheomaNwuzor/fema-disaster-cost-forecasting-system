import pandas as pd

# =========================
# LOAD DATA
# =========================

declarations = pd.read_csv("data/raw/declarations.csv")

public_assistance = pd.read_csv("data/raw/public_assistance.csv")

# =========================
# CHECK MISSING VALUES
# =========================

print("\nMissing Values Count:")
print(public_assistance.isnull().sum())

print("\nMissing Values Percentage:")
print(
    (public_assistance.isnull().sum() / len(public_assistance)) * 100
)

# =========================
# SELECT IMPORTANT COLUMNS
# =========================

declarations = declarations[
    [
        "disasterNumber",
        "state",
        "incidentType",
        "declarationDate",
        "incidentBeginDate",
        "incidentEndDate",
        "declarationType"
    ]
]

public_assistance = public_assistance[
    [
        "disasterNumber",
        "projectAmount"
    ]
]

# =========================
# REMOVE MISSING VALUES
# =========================

public_assistance = public_assistance.dropna(
    subset=["disasterNumber", "projectAmount"]
)

# =========================
# REMOVE ZERO OR NEGATIVE COSTS
# =========================

print("\nZero or Negative Project Amounts:")
print(
    (public_assistance["projectAmount"] <= 0).sum()
)

public_assistance = public_assistance[
    public_assistance["projectAmount"] > 0
]

# =========================
# AGGREGATE COSTS
# =========================

total_cost = (
    public_assistance
    .groupby("disasterNumber")["projectAmount"]
    .sum()
    .reset_index()
)

# Rename target column
total_cost.rename(
    columns={
        "projectAmount": "Total_Obligated_Amount"
    },
    inplace=True
)

# =========================
# MERGE DATASETS
# =========================

model_data = declarations.merge(
    total_cost,
    on="disasterNumber",
    how="inner"
)

# =========================
# CREATE INCIDENT DURATION
# =========================

model_data["incidentBeginDate"] = pd.to_datetime(
    model_data["incidentBeginDate"]
)

model_data["incidentEndDate"] = pd.to_datetime(
    model_data["incidentEndDate"]
)

model_data["incident_duration_days"] = (
    model_data["incidentEndDate"]
    - model_data["incidentBeginDate"]
).dt.days

# =========================
# REMOVE NEGATIVE DURATIONS
# =========================

model_data = model_data[
    model_data["incident_duration_days"] >= 0
]

# =========================
# REMOVE DUPLICATE DISASTER ROWS
# =========================

print("\nDuplicate disaster check before deduplication:")
print(model_data["disasterNumber"].value_counts().head(10))

model_data = model_data.drop_duplicates(
    subset=["disasterNumber"],
    keep="first"
)

print("\nDuplicate disaster check after deduplication:")
print(model_data["disasterNumber"].value_counts().head(10))


# =========================
# CREATE FULL DATASET
# =========================

full_dataset = model_data.copy()

# =========================
# CREATE NON-BIOLOGICAL DATASET
# =========================

non_bio_dataset = model_data[
    model_data["incidentType"] != "Biological"
]

print("\nFull Dataset Shape:")
print(full_dataset.shape)

print("\nNon-Biological Dataset Shape:")
print(non_bio_dataset.shape)

# =========================
# SAVE CLEANED DATASETS
# =========================

full_dataset.to_csv(
    "data/processed/full_dataset.csv",
    index=False
)

non_bio_dataset.to_csv(
    "data/processed/non_bio_dataset.csv",
    index=False
)


# =========================
# PRINT RESULTS
# =========================

print("\nFull Dataset Preview:")
print(full_dataset.head())

print("\nFull Dataset Shape:")
print(full_dataset.shape)

print("\nNon-Biological Dataset Shape:")
print(non_bio_dataset.shape)

print("\nColumns:")
print(full_dataset.columns.tolist())

print("\nIncident Type Counts:")
print(
    full_dataset["incidentType"]
    .value_counts()
)


