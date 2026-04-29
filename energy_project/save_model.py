import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("cleaned_energy.csv")

X = df.drop(columns=["energy"])
y = df["energy"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "energy_model.pkl")

print("✅ Model saved")
