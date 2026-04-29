import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("advanced_cleaned_social_media_comments.csv")
df = df.dropna(subset=["clean_text", "label"])

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=5000, class_weight="balanced")

param_grid = {
    "C": [0.5, 1, 3, 5, 10],
    "solver": ["liblinear", "lbfgs"]
}

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
y_prob = best_model.predict_proba(X_test_tfidf)

print("\n--- Best Parameters ---")
print(grid_search.best_params_)

print("\n--- Best CV F1 ---")
print(grid_search.best_score_)

print("\n--- Test Accuracy ---")
print(accuracy_score(y_test, y_pred))

print("\n--- Precision ---")
print(precision_score(y_test, y_pred))

print("\n--- Recall ---")
print(recall_score(y_test, y_pred))

print("\n--- F1 Score ---")
print(f1_score(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "predicted_label": y_pred,
    "negative_probability": y_prob[:, 0],
    "positive_probability": y_prob[:, 1]
})

results_df.to_csv("advanced_test_predictions.csv", index=False, encoding="utf-8-sig")

print("\nSaved as: advanced_test_predictions.csv")