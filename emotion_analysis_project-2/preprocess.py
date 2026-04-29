import pandas as pd
import re

# Turkish stopwords (lightweight custom list)
TURKISH_STOPWORDS = {
    "ve", "ile", "ama", "fakat", "de", "da", "bir", "bu", "şu", "o",
    "çok", "az", "mi", "mı", "mu", "mü", "için", "gibi", "ya", "ki",
    "ben", "sen", "o", "biz", "siz", "onlar", "şey", "daha", "en",
    "var", "yok", "olan", "olanlar", "diye", "çünkü", "hem", "ise"
}

NEGATION_WORDS = {"değil", "yok", "hiç", "asla", "degil"}

# Load dataset
df = pd.read_csv("social_media_comments (1).csv", encoding="cp1254")

# Rename columns
df = df.rename(columns={
    "Tip": "label_text",
    "Paylaşım": "text"
})

# Drop missing
df = df.dropna(subset=["text"])

# Label encoding
df["label"] = df["label_text"].map({
    "Negatif": 0,
    "Pozitif": 1
})

def normalize_repeated_characters(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def normalize_common_slang(text):
    slang_map = {
        "aq": "amk",
        "mk": "amk",
        "amq": "amk",
        "sg": "siktir git",
        "slm": "selam",
        "nbr": "ne haber",
        "ii": "iyi",
        "müko": "mükemmel",
        "rezo": "rezalet"
    }
    for slang, full in slang_map.items():
        text = re.sub(rf"\b{slang}\b", full, text)
    return text

def handle_negation(text):
    """
    Turn patterns like 'iyi değil' into 'not_iyi'
    so model can better learn negation.
    """
    words = text.split()
    new_words = []
    i = 0

    while i < len(words):
        if i < len(words) - 1 and words[i + 1] in NEGATION_WORDS:
            new_words.append("not_" + words[i])
            i += 2
        else:
            new_words.append(words[i])
            i += 1

    return " ".join(new_words)

def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in TURKISH_STOPWORDS]
    return " ".join(words)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = normalize_repeated_characters(text)
    text = normalize_common_slang(text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = handle_negation(text)
    text = remove_stopwords(text)

    return text

df["clean_text"] = df["text"].apply(clean_text)

# Remove empty cleaned rows
df = df[df["clean_text"].str.strip() != ""]

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Shape ---")
print(df.shape)

print("\n--- Missing Values ---")
print(df.isnull().sum())

df.to_csv("advanced_cleaned_social_media_comments.csv", index=False, encoding="utf-8-sig")

print("\nSaved as: advanced_cleaned_social_media_comments.csv")