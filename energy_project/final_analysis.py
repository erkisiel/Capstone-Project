import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# LOAD
# ===============================
df = pd.read_csv("advanced_cleaned_energy.csv")
model = joblib.load("final_energy_model.pkl")
features = joblib.load("energy_features.pkl")

X = df[features]
y = df["energy"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ===============================
# FINAL MODEL PREDICTIONS
# ===============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ===============================
# BASELINE MODEL
# ===============================
baseline_pred = np.full(len(y_test), y_train.mean())

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

improvement_mae = ((baseline_mae - mae) / baseline_mae) * 100
improvement_rmse = ((baseline_rmse - rmse) / baseline_rmse) * 100

# ===============================
# PREDICTION EXPORT
# ===============================
prediction_df = pd.DataFrame({
    "actual_energy": y_test.values,
    "predicted_energy": y_pred,
    "error": y_test.values - y_pred,
    "absolute_error": np.abs(y_test.values - y_pred),
    "percentage_error": (np.abs(y_test.values - y_pred) / y_test.values) * 100,
    "hour": X_test["hour"].values,
    "month": X_test["month"].values,
    "day_of_week": X_test["day_of_week"].values
})

prediction_df.to_csv("final_prediction_analysis.csv", index=False)

# ===============================
# PEAK / NON-PEAK ANALYSIS
# ===============================
peak_hours = [17, 18, 19, 20]

prediction_df["is_peak_hour"] = prediction_df["hour"].isin(peak_hours).astype(int)

peak_analysis = prediction_df.groupby("is_peak_hour").agg({
    "actual_energy": "mean",
    "predicted_energy": "mean",
    "absolute_error": "mean",
    "percentage_error": "mean"
}).reset_index()

peak_analysis["period"] = peak_analysis["is_peak_hour"].map({
    0: "Non-Peak",
    1: "Peak"
})

peak_analysis = peak_analysis[
    ["period", "actual_energy", "predicted_energy", "absolute_error", "percentage_error"]
]

peak_analysis.to_csv("peak_non_peak_analysis.csv", index=False)

# ===============================
# ERROR BY HOUR
# ===============================
error_by_hour = prediction_df.groupby("hour").agg({
    "absolute_error": "mean",
    "percentage_error": "mean"
}).reset_index()

error_by_hour.to_csv("error_by_hour_final.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(error_by_hour["hour"], error_by_hour["absolute_error"])
plt.title("Average Prediction Error by Hour")
plt.xlabel("Hour")
plt.ylabel("Mean Absolute Error")
plt.tight_layout()
plt.savefig("error_by_hour_final.png")
plt.close()

# ===============================
# ERROR BY MONTH
# ===============================
error_by_month = prediction_df.groupby("month").agg({
    "absolute_error": "mean",
    "percentage_error": "mean"
}).reset_index()

error_by_month.to_csv("error_by_month_final.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(error_by_month["month"], error_by_month["absolute_error"])
plt.title("Average Prediction Error by Month")
plt.xlabel("Month")
plt.ylabel("Mean Absolute Error")
plt.tight_layout()
plt.savefig("error_by_month_final.png")
plt.close()

# ===============================
# BASELINE VS FINAL MODEL CHART
# ===============================
comparison = pd.DataFrame({
    "Model": ["Baseline Mean Model", "Final Tuned Decision Tree"],
    "MAE": [baseline_mae, mae],
    "RMSE": [baseline_rmse, rmse],
    "R2": [baseline_r2, r2]
})

comparison.to_csv("baseline_vs_final_model.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["MAE"])
plt.title("Baseline vs Final Model MAE")
plt.ylabel("MAE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("baseline_vs_final_mae.png")
plt.close()

# ===============================
# FEATURE IMPORTANCE SUMMARY
# ===============================
if hasattr(model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance.to_csv("final_feature_importance_detailed.csv", index=False)

# ===============================
# REPORT READY SUMMARY
# ===============================
summary = pd.DataFrame({
    "Item": [
        "Dataset Rows After Advanced Preprocessing",
        "Dataset Columns After Advanced Preprocessing",
        "Missing Values",
        "Final Model",
        "Final Test MAE",
        "Final Test RMSE",
        "Final Test R2",
        "Baseline MAE",
        "Baseline RMSE",
        "Baseline R2",
        "MAE Improvement Over Baseline (%)",
        "RMSE Improvement Over Baseline (%)",
        "Mean Percentage Error (%)"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        int(df.isnull().sum().sum()),
        "Tuned Decision Tree Regressor",
        round(mae, 4),
        round(rmse, 4),
        round(r2, 4),
        round(baseline_mae, 4),
        round(baseline_rmse, 4),
        round(baseline_r2, 4),
        round(improvement_mae, 2),
        round(improvement_rmse, 2),
        round(prediction_df["percentage_error"].mean(), 4)
    ]
})

summary.to_csv("final_report_summary.csv", index=False)

# ===============================
# PRINT RESULTS
# ===============================
print("\n===== FINAL MODEL PERFORMANCE =====")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

print("\n===== BASELINE PERFORMANCE =====")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline R2: {baseline_r2:.4f}")

print("\n===== IMPROVEMENT OVER BASELINE =====")
print(f"MAE Improvement: {improvement_mae:.2f}%")
print(f"RMSE Improvement: {improvement_rmse:.2f}%")

print("\n===== MEAN PERCENTAGE ERROR =====")
print(f"{prediction_df['percentage_error'].mean():.4f}%")

print("\n✅ Final analysis completed")
print("Saved files:")
print("- final_prediction_analysis.csv")
print("- peak_non_peak_analysis.csv")
print("- error_by_hour_final.csv")
print("- error_by_month_final.csv")
print("- error_by_hour_final.png")
print("- error_by_month_final.png")
print("- baseline_vs_final_model.csv")
print("- baseline_vs_final_mae.png")
print("- final_report_summary.csv")
print("- final_feature_importance_detailed.csv")
