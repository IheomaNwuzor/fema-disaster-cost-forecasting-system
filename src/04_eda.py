import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATASETS
# =========================

full_df = pd.read_csv(
    "data/processed/full_dataset.csv"
)

non_bio_df = pd.read_csv(
    "data/processed/non_bio_dataset.csv"
)

# =========================
# BASIC COMPARISON
# =========================

print("\nFULL DATASET SHAPE:")
print(full_df.shape)

print("\nNON-BIOLOGICAL DATASET SHAPE:")
print(non_bio_df.shape)

# =========================
# SUMMARY STATISTICS
# =========================

print("\nFULL DATASET SUMMARY:")
print(
    full_df[
        [
            "Total_Obligated_Amount",
            "incident_duration_days"
        ]
    ].describe()
)

print("\nNON-BIOLOGICAL DATASET SUMMARY:")
print(
    non_bio_df[
        [
            "Total_Obligated_Amount",
            "incident_duration_days"
        ]
    ].describe()
)

# =========================
# COST DISTRIBUTION
# =========================

plt.figure(figsize=(10, 5))

plt.hist(
    full_df["Total_Obligated_Amount"],
    bins=30,
    alpha=0.6,
    label="Full Dataset"
)

plt.hist(
    non_bio_df["Total_Obligated_Amount"],
    bins=30,
    alpha=0.6,
    label="Non-Biological Dataset"
)

plt.title("Cost Distribution Comparison")

plt.xlabel("Total Obligated Amount")

plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/cost_distribution_comparison.png"
)

plt.close()

# =========================
# DURATION DISTRIBUTION
# =========================

plt.figure(figsize=(10, 5))

plt.hist(
    full_df["incident_duration_days"],
    bins=30,
    alpha=0.6,
    label="Full Dataset"
)

plt.hist(
    non_bio_df["incident_duration_days"],
    bins=30,
    alpha=0.6,
    label="Non-Biological Dataset"
)

plt.title("Duration Distribution Comparison")

plt.xlabel("Incident Duration Days")

plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/duration_distribution_comparison.png"
)

plt.close()

# =========================
# SCATTERPLOT
# =========================

plt.figure(figsize=(10, 5))

plt.scatter(
    full_df["incident_duration_days"],
    full_df["Total_Obligated_Amount"],
    alpha=0.5,
    label="Full Dataset"
)

plt.title("Duration vs Cost")

plt.xlabel("Incident Duration Days")

plt.ylabel("Total Obligated Amount")

plt.tight_layout()

plt.savefig(
    "reports/duration_vs_cost.png"
)

plt.close()

# =========================
# TOP COSTLY INCIDENT TYPES
# =========================

print("\nAVERAGE COST BY INCIDENT TYPE:")

print(
    full_df.groupby("incidentType")[
        "Total_Obligated_Amount"
    ]
    .mean()
    .sort_values(ascending=False)
)