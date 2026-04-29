import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
df = pd.read_csv("cleaned_social_media_comments.csv")
df = df.dropna(subset=["clean_text", "label"])

X = df["clean_text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=2000, C=5, solver="liblinear")
model.fit(X_train_tfidf, y_train)

# Get feature names and coefficients
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

# Top keywords
top_positive_indices = np.argsort(coefficients)[-20:]
top_negative_indices = np.argsort(coefficients)[:20]

positive_words = pd.DataFrame({
    "Keyword": feature_names[top_positive_indices],
    "Coefficient": coefficients[top_positive_indices]
}).sort_values(by="Coefficient", ascending=False)

negative_words = pd.DataFrame({
    "Keyword": feature_names[top_negative_indices],
    "Coefficient": coefficients[top_negative_indices]
}).sort_values(by="Coefficient", ascending=True)

print("\n--- Top 20 Positive Keywords ---")
print(positive_words)

print("\n--- Top 20 Negative Keywords ---")
print(negative_words)

# Save to CSV
positive_words.to_csv("top_positive_keywords.csv", index=False, encoding="utf-8-sig")
negative_words.to_csv("top_negative_keywords.csv", index=False, encoding="utf-8-sig")

print("\nSaved files:")
print("- top_positive_keywords.csv")
print("- top_negative_keywords.csv")