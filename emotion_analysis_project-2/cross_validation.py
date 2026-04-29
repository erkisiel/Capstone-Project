import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load cleaned dataset
df = pd.read_csv("cleaned_social_media_comments.csv")
df = df.dropna(subset=["clean_text", "label"])

X = df["clean_text"]
y = df["label"]

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
    ("model", LogisticRegression(max_iter=2000, C=1, solver="liblinear"))
])

# Cross-validation
accuracy_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
f1_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")

print("\n--- Cross-Validation Accuracy Scores ---")
print(accuracy_scores)
print("Mean Accuracy:", accuracy_scores.mean())

print("\n--- Cross-Validation F1 Scores ---")
print(f1_scores)
print("Mean F1 Score:", f1_scores.mean())