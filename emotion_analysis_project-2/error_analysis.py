import pandas as pd

df = pd.read_csv("advanced_test_predictions.csv")

errors = df[df["true_label"] != df["predicted_label"]].copy()

label_map = {0: "Negative", 1: "Positive"}
errors["true_label_text"] = errors["true_label"].map(label_map)
errors["predicted_label_text"] = errors["predicted_label"].map(label_map)

errors["confidence"] = errors[["negative_probability", "positive_probability"]].max(axis=1)
errors = errors.sort_values(by="confidence", ascending=False)

print("\n--- Top 20 Wrong Predictions ---")
print(errors[["text", "true_label_text", "predicted_label_text", "negative_probability", "positive_probability"]].head(20))

errors.to_csv("model_errors.csv", index=False, encoding="utf-8-sig")
print("\nSaved as: model_errors.csv")