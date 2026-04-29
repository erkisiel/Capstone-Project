import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load original dataset
df = pd.read_csv("3607643.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")
df = df.rename(columns={"DAYTON_MW": "energy"})

# Time features
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month
df["year"] = df["Datetime"].dt.year
df["day_of_week"] = df["Datetime"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Advanced lag features
df["lag_1"] = df["energy"].shift(1)
df["lag_24"] = df["energy"].shift(24)
df["lag_168"] = df["energy"].shift(168)

df["rolling_24"] = df["energy"].rolling(window=24).mean()
df["rolling_168"] = df["energy"].rolling(window=168).mean()

df = df.dropna()

features = [
    "hour", "day", "month", "year", "day_of_week", "is_weekend",
    "lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168"
]

X = df[features]
y = df["energy"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Tuned Decision Tree to reduce overfitting
model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("===== Tuned Decision Tree =====")
print(f"Train MAE: {train_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Train R2: {train_r2:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R2: {test_r2:.4f}")

# Save files
joblib.dump(model, "final_energy_model.pkl")
joblib.dump(features, "energy_features.pkl")

df.to_csv("advanced_cleaned_energy.csv", index=False)

pred_df = pd.DataFrame({
    "actual_energy": y_test.values,
    "predicted_energy": test_pred,
    "error": y_test.values - test_pred,
    "absolute_error": np.abs(y_test.values - test_pred)
})
pred_df.to_csv("final_energy_predictions.csv", index=False)

# Actual vs predicted line graph
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:500], label="Actual")
plt.plot(test_pred[:500], label="Predicted")
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Time Index")
plt.ylabel("Energy Consumption (MW)")
plt.legend()
plt.tight_layout()
plt.savefig("final_actual_vs_predicted_line.png")
plt.close()

# Feature importance
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("final_feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.title("Final Model Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("final_feature_importance.png")
plt.close()

# Error distribution
plt.figure(figsize=(8, 5))
plt.hist(pred_df["error"], bins=40)
plt.title("Final Model Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("final_error_distribution.png")
plt.close()

print("\n✅ Tuned final model saved")
print("Saved:")
print("- final_energy_model.pkl")
print("- advanced_cleaned_energy.csv")
print("- final_energy_predictions.csv")
print("- final_actual_vs_predicted_line.png")
print("- final_feature_importance.png")
print("- final_error_distribution.png")
