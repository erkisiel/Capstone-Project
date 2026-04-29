import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load original data
df = pd.read_csv("3607643.csv")

# Date conversion
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

# Rename target
df = df.rename(columns={"DAYTON_MW": "energy"})

# Basic time features
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month
df["year"] = df["Datetime"].dt.year
df["day_of_week"] = df["Datetime"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Advanced time-series features
df["lag_1"] = df["energy"].shift(1)
df["lag_24"] = df["energy"].shift(24)
df["lag_168"] = df["energy"].shift(168)

df["rolling_24"] = df["energy"].rolling(window=24).mean()
df["rolling_168"] = df["energy"].rolling(window=168).mean()

# Remove rows created as NaN because of lag/rolling
df = df.dropna()

features = [
    "hour", "day", "month", "year", "day_of_week", "is_weekend",
    "lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168"
]

X = df[features]
y = df["energy"]

# Chronological split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []
best_model = None
best_model_name = ""
best_r2 = -999
best_predictions = None

for name, model in models.items():
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)

    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    results.append({
        "Model": name,
        "Train MAE": train_mae,
        "Train RMSE": train_rmse,
        "Train R2": train_r2,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Test R2": test_r2
    })

    print(f"\n===== {name} =====")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    if test_r2 > best_r2:
        best_r2 = test_r2
        best_model = model
        best_model_name = name
        best_predictions = test_preds

results_df = pd.DataFrame(results)
results_df.to_csv("advanced_energy_model_results.csv", index=False)

joblib.dump(best_model, "advanced_energy_model.pkl")
joblib.dump(features, "energy_features.pkl")

prediction_df = pd.DataFrame({
    "actual_energy": y_test.values,
    "predicted_energy": best_predictions,
    "error": y_test.values - best_predictions
})
prediction_df.to_csv("energy_predictions.csv", index=False)

# Save final cleaned advanced dataset
df.to_csv("advanced_cleaned_energy.csv", index=False)

# Model comparison graph
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Test R2"])
plt.title("Model Comparison by Test R2 Score")
plt.xlabel("Model")
plt.ylabel("Test R2 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("energy_model_comparison.png")
plt.close()

# Actual vs predicted scatter
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, alpha=0.5)
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.tight_layout()
plt.savefig("actual_vs_predicted_energy.png")
plt.close()

# Error distribution
errors = y_test.values - best_predictions
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=40)
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("energy_error_distribution.png")
plt.close()

# Time-series comparison
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:500], label="Actual")
plt.plot(best_predictions[:500], label="Predicted")
plt.title("Actual vs Predicted Energy Over Time")
plt.xlabel("Time Index")
plt.ylabel("Energy Consumption")
plt.legend()
plt.tight_layout()
plt.savefig("time_series_actual_vs_predicted.png")
plt.close()

# Feature importance
if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv("feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

# Error analysis by hour
error_analysis_df = pd.DataFrame({
    "hour": X_test["hour"].values,
    "actual_energy": y_test.values,
    "predicted_energy": best_predictions,
    "absolute_error": np.abs(y_test.values - best_predictions)
})

hour_error = error_analysis_df.groupby("hour")["absolute_error"].mean().reset_index()
hour_error.to_csv("error_by_hour.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(hour_error["hour"], hour_error["absolute_error"])
plt.title("Average Prediction Error by Hour")
plt.xlabel("Hour")
plt.ylabel("Mean Absolute Error")
plt.tight_layout()
plt.savefig("error_by_hour.png")
plt.close()

print("\n✅ Advanced training completed")
print(f"Best Model: {best_model_name}")
print("Saved files:")
print("- advanced_energy_model.pkl")
print("- energy_features.pkl")
print("- advanced_energy_model_results.csv")
print("- energy_predictions.csv")
print("- advanced_cleaned_energy.csv")
print("- energy_model_comparison.png")
print("- actual_vs_predicted_energy.png")
print("- energy_error_distribution.png")
print("- time_series_actual_vs_predicted.png")
print("- feature_importance.png")
print("- error_by_hour.png")
