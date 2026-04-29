import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned dataset
df = pd.read_csv("cleaned_social_media_comments.csv")
df = df.dropna(subset=["clean_text", "label"])

# Features and labels
X = df["clean_text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
lr = LogisticRegression(max_iter=2000)

# Hyperparameter grid
param_grid = {
    "C": [0.1, 1, 5, 10],
    "solver": ["liblinear", "lbfgs"]
}

# Grid search
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_tfidf)

print("\n--- Best Parameters ---")
print(grid_search.best_params_)

print("\n--- Best Cross-Validation Score ---")
print(grid_search.best_score_)

print("\n--- Test Accuracy ---")
print(accuracy_score(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))