import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix

# -----------------------------
# LEXICONS
# -----------------------------
NEGATIVE_WORDS = {
    "kötü", "berbat", "rezalet", "iğrenç", "nefret", "aptal", "salak",
    "gerzek", "embesil", "amk", "orospu", "geber", "öl", "öldürürüm",
    "haketmiyorsun", "pezevenk", "sokayım", "götlek", "düşman"
}

POSITIVE_WORDS = {
    "güzel", "harika", "mükemmel", "mutlu", "seviyorum", "başarılı",
    "tebrikler", "hayırlı", "iyi", "süper", "güzelsin", "mutluyum",
    "tatlısın", "teşekkürler", "başardım", "sevgi", "düğün"
}

PROFANITY_WORDS = {
    "amk", "orospu", "pezevenk", "sikeyim", "sokayım", "göt",
    "salak", "aptal", "gerzek", "embesil"
}

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("social_media_comments (1).csv", encoding="cp1254")

df = df.rename(columns={
    "Tip": "label_text",
    "Paylaşım": "text"
})

df = df.dropna(subset=["text"])

df["label"] = df["label_text"].map({
    "Negatif": 0,
    "Pozitif": 1
})

# -----------------------------
# TEXT NORMALIZATION
# -----------------------------
def normalize_repeated_characters(text):
    return re.sub(r"(.)\1{2,}", r"\1", text)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = normalize_repeated_characters(text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s!]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    replacements = {
        "kotu": "kötü",
        "guzel": "güzel",
        "cok": "çok",
        "berbaat": "berbat",
        "rezalett": "rezalet",
        "iyii": "iyi",
        "iyiim": "iyi"
    }

    words = text.split()
    words = [replacements.get(w, w) for w in words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def extract_features(original_text, cleaned_text):
    words = cleaned_text.split()

    text_length = len(original_text)
    word_count = len(words)
    exclamation_count = original_text.count("!")
    uppercase_ratio = sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1)
    positive_word_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_word_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    profanity_count = sum(1 for w in words if w in PROFANITY_WORDS)
    repeated_char_count = len(re.findall(r"(.)\1{2,}", original_text.lower()))

    return [
        text_length,
        word_count,
        exclamation_count,
        uppercase_ratio,
        positive_word_count,
        negative_word_count,
        profanity_count,
        repeated_char_count
    ]

extra_features = np.array([
    extract_features(orig, clean)
    for orig, clean in zip(df["text"], df["clean_text"])
])

# -----------------------------
# SPLIT DATA
# -----------------------------
X_text = df["clean_text"]
y = df["label"]

X_train_text, X_test_text, X_train_extra, X_test_extra, y_train, y_test = train_test_split(
    X_text,
    extra_features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Convert extra features to sparse
X_train_extra_sparse = csr_matrix(X_train_extra)
X_test_extra_sparse = csr_matrix(X_test_extra)

# Combine TF-IDF + engineered features
X_train_final = hstack([X_train_tfidf, X_train_extra_sparse])
X_test_final = hstack([X_test_tfidf, X_test_extra_sparse])

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(
    max_iter=5000,
    C=5,
    solver="lbfgs",
    class_weight="balanced"
)

model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

# -----------------------------
# RESULTS
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- FEATURE-ENHANCED MODEL RESULTS ---")
print(f"Accuracy: {accuracy:.6f}")
print(f"F1 Score: {f1:.6f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Save comparison-friendly output
results_df = pd.DataFrame([{
    "Model": "Feature-Enhanced Logistic Regression",
    "Accuracy": accuracy,
    "F1 Score": f1
}])

results_df.to_csv("feature_enhanced_results.csv", index=False, encoding="utf-8-sig")

print("\nSaved as: feature_enhanced_results.csv")