import pandas as pd

# Load dataset
df = pd.read_csv("social_media_comments (1).csv", encoding="cp1254")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Class Distribution ---")
print(df["Tip"].value_counts())

print("\n--- Data Types ---")
print(df.dtypes)