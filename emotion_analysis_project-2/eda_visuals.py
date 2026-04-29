import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("social_media_comments (1).csv", encoding="cp1254")

# Rename columns
df = df.rename(columns={
    "Tip": "label_text",
    "Paylaşım": "text"
})

# Missing values chart
missing_counts = df.isnull().sum()

plt.figure(figsize=(6, 4))
plt.bar(missing_counts.index, missing_counts.values)
plt.title("Missing Values per Column")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("missing_values_chart.png")
plt.close()

# Class distribution chart
class_counts = df["label_text"].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(class_counts.index, class_counts.values)
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

# Text length distribution
df = df.dropna(subset=["text"])
df["text_length"] = df["text"].astype(str).apply(len)

plt.figure(figsize=(8, 5))
plt.hist(df["text_length"], bins=30)
plt.title("Text Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("text_length_distribution.png")
plt.close()

print("EDA charts created successfully.")
print("- missing_values_chart.png")
print("- class_distribution.png")
print("- text_length_distribution.png")