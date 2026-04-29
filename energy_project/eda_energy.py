import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_energy.csv")

# enerji dağılımı
plt.figure()
df["energy"].hist()
plt.savefig("energy_hist.png")

# saat vs enerji
plt.figure()
plt.scatter(df["hour"], df["energy"])
plt.savefig("hour_vs_energy.png")

print("✅ EDA DONE")
