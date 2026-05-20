import pandas as pd

declarations = pd.read_csv("data/raw/declarations.csv")
public_assistance = pd.read_csv("data/raw/public_assistance.csv")
disaster_summaries = pd.read_csv("data/raw/disaster_summaries.csv")

print("Declarations:", declarations.shape)
print("Public assistance:", public_assistance.shape)
print("Disaster summaries:", disaster_summaries.shape)

print("\nDeclarations columns:")
print(declarations.columns.tolist())

print("\nPublic assistance columns:")
print(public_assistance.columns.tolist())

print("\nDisaster summaries columns:")
print(disaster_summaries.columns.tolist())