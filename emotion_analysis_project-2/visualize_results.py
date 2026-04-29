import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from wordcloud import WordCloud

# Load cleaned dataset
df = pd.read_csv("cleaned_social_media_comments.csv")

# Drop missing values
df = df.dropna(subset=["clean_text", "label"])

# Features and labels
X = df["clean_text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png", bbox_inches="tight")
    plt.close()

# Save results table
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

# Accuracy comparison chart
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.close()

# F1-score comparison chart
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["F1-Score"])
plt.title("Model F1-Score Comparison")
plt.ylabel("F1-Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("f1_comparison.png")
plt.close()

# Positive word cloud
positive_text = " ".join(df[df["label"] == 1]["clean_text"].astype(str))
wordcloud_pos = WordCloud(width=1000, height=500, background_color="white").generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Positive Comments")
plt.tight_layout()
plt.savefig("positive_wordcloud.png")
plt.close()

# Negative word cloud
negative_text = " ".join(df[df["label"] == 0]["clean_text"].astype(str))
wordcloud_neg = WordCloud(width=1000, height=500, background_color="white").generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Negative Comments")
plt.tight_layout()
plt.savefig("negative_wordcloud.png")
plt.close()

print("\nVisualizations created successfully.")
print("Saved files:")
print("- model_results.csv")
print("- accuracy_comparison.png")
print("- f1_comparison.png")
print("- logistic_regression_confusion_matrix.png")
print("- naive_bayes_confusion_matrix.png")
print("- linear_svm_confusion_matrix.png")
print("- positive_wordcloud.png")
print("- negative_wordcloud.png")