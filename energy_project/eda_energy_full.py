import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("advanced_cleaned_energy.csv")

print("--- Dataset Shape ---")
print(df.shape)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe())

# Missing values chart
missing = df.isnull().sum()
plt.figure(figsize=(8, 4))
missing.plot(kind="bar")
plt.title("Missing Values by Column")
plt.xlabel("Columns")
plt.ylabel("Missing Count")
plt.tight_layout()
plt.savefig("missing_values_energy.png")
plt.close()

# Target distribution
plt.figure(figsize=(8, 5))
df["energy"].hist(bins=50)
plt.title("Energy Consumption Distribution")
plt.xlabel("Energy Consumption (MW)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("energy_distribution_full.png")
plt.close()

# Boxplot for outlier view
plt.figure(figsize=(8, 5))
plt.boxplot(df["energy"])
plt.title("Energy Consumption Boxplot")
plt.ylabel("Energy Consumption (MW)")
plt.tight_layout()
plt.savefig("energy_boxplot.png")
plt.close()

# Average energy by hour
plt.figure(figsize=(8, 5))
df.groupby("hour")["energy"].mean().plot(kind="bar")
plt.title("Average Energy Consumption by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Energy Consumption (MW)")
plt.tight_layout()
plt.savefig("avg_energy_by_hour_full.png")
plt.close()

# Average energy by day of week
plt.figure(figsize=(8, 5))
df.groupby("day_of_week")["energy"].mean().plot(kind="bar")
plt.title("Average Energy Consumption by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Energy Consumption (MW)")
plt.tight_layout()
plt.savefig("avg_energy_by_day_of_week.png")
plt.close()

# Average energy by month
plt.figure(figsize=(8, 5))
df.groupby("month")["energy"].mean().plot(kind="bar")
plt.title("Average Energy Consumption by Month")
plt.xlabel("Month")
plt.ylabel("Average Energy Consumption (MW)")
plt.tight_layout()
plt.savefig("avg_energy_by_month_full.png")
plt.close()

# Correlation matrix
corr = df.drop(columns=["Datetime"], errors="ignore").corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix_energy.png")
plt.close()

# Time-series trend sample
plt.figure(figsize=(12, 5))
df["energy"].iloc[:1000].plot()
plt.title("Energy Consumption Time Series Sample")
plt.xlabel("Time Index")
plt.ylabel("Energy Consumption (MW)")
plt.tight_layout()
plt.savefig("energy_time_series_sample.png")
plt.close()

print("\n✅ Full EDA completed")
print("Saved visuals:")
print("- missing_values_energy.png")
print("- energy_distribution_full.png")
print("- energy_boxplot.png")
print("- avg_energy_by_hour_full.png")
print("- avg_energy_by_day_of_week.png")
print("- avg_energy_by_month_full.png")
print("- correlation_matrix_energy.png")
print("- energy_time_series_sample.png")
