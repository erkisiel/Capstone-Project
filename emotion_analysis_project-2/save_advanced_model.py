import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load advanced cleaned dataset
df = pd.read_csv("advanced_cleaned_social_media_comments.csv")
df = df.dropna(subset=["clean_text", "label"])

X = df["clean_text"]
y = df["label"]

# Better TF-IDF
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(X)

# Final advanced model
model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    C=5,
    solver="lbfgs"
)

model.fit(X_tfidf, y)

joblib.dump(model, "advanced_sentiment_model.pkl")
joblib.dump(vectorizer, "advanced_tfidf_vectorizer.pkl")

print("Saved:")
print("- advanced_sentiment_model.pkl")
print("- advanced_tfidf_vectorizer.pkl")