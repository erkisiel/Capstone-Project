import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
df = pd.read_csv("cleaned_social_media_comments.csv")
df = df.dropna(subset=["clean_text", "label"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

X = df["clean_text"]
y = df["label"]

# Vectorizer and model
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=2000, C=1, solver="liblinear")
model.fit(X_tfidf, y)

# User input
user_text = input("Enter a comment: ")
cleaned_input = clean_text(user_text)
input_vector = vectorizer.transform([cleaned_input])

prediction = model.predict(input_vector)[0]
probability = model.predict_proba(input_vector)[0]

if prediction == 1:
    print("\nPredicted Sentiment: Positive")
else:
    print("\nPredicted Sentiment: Negative")

print(f"Negative Probability: {probability[0]:.4f}")
print(f"Positive Probability: {probability[1]:.4f}")