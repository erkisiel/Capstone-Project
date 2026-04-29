import pandas as pd

df = pd.read_csv("3607643.csv")

# datetime dönüşümü
df["Datetime"] = pd.to_datetime(df["Datetime"])

# feature engineering
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month
df["year"] = df["Datetime"].dt.year
df["day_of_week"] = df["Datetime"].dt.dayofweek

# target rename
df = df.rename(columns={"DAYTON_MW": "energy"})

# datetime kaldır
df = df.drop(columns=["Datetime"])

# kaydet
df.to_csv("cleaned_energy.csv", index=False)

print("✅ Preprocessing DONE")
print(df.head())
