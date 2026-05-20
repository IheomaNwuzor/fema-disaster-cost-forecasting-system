import pandas as pd

# =========================
# LOAD DATA
# =========================

public_assistance = pd.read_csv(
    "data/raw/public_assistance.csv"
)

# =========================
# BASIC INFO
# =========================

print("\nDataset Shape:")
print(public_assistance.shape)

print("\nColumns:")
print(public_assistance.columns.tolist())

# =========================
# MISSING VALUE COUNT
# =========================

print("\nMissing Values Count:")
print(public_assistance.isnull().sum())

# =========================
# MISSING VALUE PERCENTAGE
# =========================

missing_percentage = (
    public_assistance.isnull().sum()
    / len(public_assistance)
) * 100

print("\nMissing Values Percentage:")
print(missing_percentage)

# =========================
# SORT NULL PERCENTAGES
# =========================

print("\nSorted Missing Value Percentages:")
print(
    missing_percentage.sort_values(
        ascending=False
    )
)