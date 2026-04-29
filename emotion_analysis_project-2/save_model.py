import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
df = pd.read_csv("cleaned_social_media_comments.csv")
df = df.dropna(subset=["clean_text", "label"])

X = df["clean_text"]
y = df["label"]

# Train vectorizer
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# Train final model
model = LogisticRegression(max_iter=2000, C=5, solver="liblinear")
model.fit(X_tfidf, y)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model saved as sentiment_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")