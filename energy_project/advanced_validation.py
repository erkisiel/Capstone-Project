import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("advanced_cleaned_energy.csv")
features = joblib.load("energy_features.pkl")

X = df[features]
y = df["energy"]

# ===============================
# TIME SERIES CROSS VALIDATION
# ===============================
tscv = TimeSeriesSplit(n_splits=5)

cv_results = []

fold = 1

for train_index, test_index in tscv.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    cv_results.append({
        "Fold": fold,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n===== Fold {fold} =====")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    fold += 1

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("time_series_cv_results.csv", index=False)

print("\n===== TIME SERIES CV AVERAGE =====")
print(f"Average MAE: {cv_df['MAE'].mean():.4f}")
print(f"Average RMSE: {cv_df['RMSE'].mean():.4f}")
print(f"Average R2: {cv_df['R2'].mean():.4f}")

# CV R2 graph
plt.figure(figsize=(8, 5))
plt.plot(cv_df["Fold"], cv_df["R2"], marker="o")
plt.title("Time-Series Cross Validation R2 Scores")
plt.xlabel("Fold")
plt.ylabel("R2 Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("time_series_cv_r2.png")
plt.close()

# ===============================
# FINAL MODEL RESIDUAL ANALYSIS
# ===============================
final_model = joblib.load("final_energy_model.pkl")

split = int(len(df) * 0.8)

X_test_final = X.iloc[split:]
y_test_final = y.iloc[split:]

final_preds = final_model.predict(X_test_final)

residuals = y_test_final.values - final_preds

residual_df = pd.DataFrame({
    "actual_energy": y_test_final.values,
    "predicted_energy": final_preds,
    "residual": residuals,
    "absolute_residual": np.abs(residuals)
})

residual_df.to_csv("residual_analysis.csv", index=False)

# Residual histogram
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("residual_distribution.png")
plt.close()

# Residual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(final_preds, residuals, alpha=0.4)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Energy")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig("residual_vs_predicted.png")
plt.close()

# ===============================
# AUTOCORRELATION ANALYSIS
# ===============================
try:
    from pandas.plotting import autocorrelation_plot

    plt.figure(figsize=(10, 5))
    autocorrelation_plot(df["energy"].iloc[:5000])
    plt.title("Energy Consumption Autocorrelation Plot")
    plt.tight_layout()
    plt.savefig("autocorrelation_energy.png")
    plt.close()

    print("\n✅ Autocorrelation plot saved")

except Exception as e:
    print("\nAutocorrelation plot could not be created:")
    print(e)

# ===============================
# SHAP ANALYSIS OPTIONAL
# ===============================
try:
    import shap

    sample_X = X_test_final.sample(min(1000, len(X_test_final)), random_state=42)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(sample_X)

    shap.summary_plot(shap_values, sample_X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    plt.close()

    print("\n✅ SHAP summary plot saved")

except Exception as e:
    print("\nSHAP could not be created.")
    print("If you want SHAP, install it with:")
    print("pip install shap")
    print("Error:")
    print(e)

# ===============================
# ADVANCED SUMMARY
# ===============================
advanced_summary = pd.DataFrame({
    "Item": [
        "TimeSeriesSplit Average MAE",
        "TimeSeriesSplit Average RMSE",
        "TimeSeriesSplit Average R2",
        "Residual Mean",
        "Residual Std",
        "Mean Absolute Residual",
        "Max Absolute Residual"
    ],
    "Value": [
        round(cv_df["MAE"].mean(), 4),
        round(cv_df["RMSE"].mean(), 4),
        round(cv_df["R2"].mean(), 4),
        round(residual_df["residual"].mean(), 4),
        round(residual_df["residual"].std(), 4),
        round(residual_df["absolute_residual"].mean(), 4),
        round(residual_df["absolute_residual"].max(), 4)
    ]
})

advanced_summary.to_csv("advanced_validation_summary.csv", index=False)

print("\n✅ Advanced validation completed")
print("Saved files:")
print("- time_series_cv_results.csv")
print("- time_series_cv_r2.png")
print("- residual_analysis.csv")
print("- residual_distribution.png")
print("- residual_vs_predicted.png")
print("- autocorrelation_energy.png")
print("- shap_summary_plot.png if SHAP installed")
print("- advanced_validation_summary.csv")
