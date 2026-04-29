import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="MEF Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
/* ---------- APP ---------- */
.stApp {
    background: linear-gradient(180deg, #f3f6fb 0%, #edf2f8 100%);
}

/* ---------- GLOBAL TEXT ---------- */
html, body {
    color: #111827 !important;
}

p, li, label, span {
    color: #111827 !important;
}

/* ---------- MAIN CONTENT ---------- */
section.main > div {
    padding-top: 1rem;
}

.main .block-container,
.main .block-container p,
.main .block-container div,
.main .block-container span,
.main .block-container label,
.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.main .block-container h4,
.main .block-container h5,
.main .block-container h6 {
    color: #111827 !important;
}

/* ---------- HERO ---------- */
.hero-box {
    background: linear-gradient(135deg, #05070b 0%, #0f172a 55%, #1f3b63 100%);
    border-radius: 26px;
    padding: 28px 34px;
    box-shadow: 0 14px 32px rgba(0,0,0,0.18);
    margin-bottom: 18px;
}

.hero-title {
    color: #ffffff !important;
    font-size: 2.35rem;
    font-weight: 800;
    margin-bottom: 6px;
}

.hero-subtitle {
    color: #e5e7eb !important;
    font-size: 1rem;
    margin-bottom: 0;
}

/* ---------- TOP CARDS ---------- */
.top-card {
    background: white;
    border: 1px solid #d9e1eb;
    border-radius: 20px;
    padding: 16px;
    box-shadow: 0 8px 18px rgba(17,24,39,0.06);
    min-height: 115px;
}

.top-card-title {
    color: #0f172a !important;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.top-card-text {
    color: #475569 !important;
    font-size: 0.9rem;
    line-height: 1.4;
}

/* ---------- PANELS ---------- */
.panel-card {
    background: rgba(255,255,255,0.98);
    border: 1px solid #d8e0ea;
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 10px 24px rgba(17,24,39,0.06);
    margin-bottom: 18px;
}

.panel-card,
.panel-card p,
.panel-card div,
.panel-card span,
.panel-card label,
.panel-card h1,
.panel-card h2,
.panel-card h3,
.panel-card h4,
.panel-card h5,
.panel-card h6 {
    color: #111827 !important;
}

.panel-title {
    color: #0f172a !important;
    font-size: 1.3rem;
    font-weight: 800;
    margin-bottom: 14px;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05070b 0%, #0f172a 60%, #1f2937 100%);
    border-right: 2px solid #253243;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    color: #f8fafc !important;
}

/* ---------- BUTTONS ---------- */
div.stButton > button {
    background: linear-gradient(135deg, #0f172a 0%, #1f3b63 100%);
    color: white !important;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    box-shadow: 0 8px 18px rgba(15,23,42,0.16);
}

div.stButton > button * {
    color: white !important;
    fill: white !important;
}

/* ---------- DOWNLOAD BUTTONS ---------- */
div.stDownloadButton > button {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    color: white !important;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1rem;
    font-weight: 700;
}

div.stDownloadButton > button * {
    color: white !important;
    fill: white !important;
}

/* ---------- TABS ---------- */
[data-baseweb="tab-list"] {
    gap: 8px;
}

button[role="tab"] {
    border-radius: 14px !important;
    background: #e8edf4 !important;
    color: #0f172a !important;
    padding: 10px 16px !important;
    font-weight: 700 !important;
    border: 1px solid #d8e0ea !important;
}

button[role="tab"] * {
    color: inherit !important;
}

button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #05070b 0%, #0f172a 60%, #1f3b63 100%) !important;
    color: white !important;
    border: none !important;
}

button[role="tab"][aria-selected="true"] * {
    color: white !important;
}

/* ---------- TEXT AREA ---------- */
[data-testid="stTextArea"] label {
    color: #111827 !important;
    font-weight: 600;
}

[data-testid="stTextArea"] textarea {
    border-radius: 16px !important;
    border: 1.5px solid #d4dbe7 !important;
    background: #ffffff !important;
    color: #111827 !important;
    box-shadow: none !important;
}

[data-testid="stTextArea"] textarea:focus {
    border: 1.5px solid #94a3b8 !important;
    box-shadow: 0 0 0 1px #94a3b8 !important;
    outline: none !important;
}

/* ---------- FILE UPLOADER ---------- */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border-radius: 16px !important;
    padding: 8px !important;
    border: 1px solid #d4dbe7 !important;
    box-shadow: 0 6px 16px rgba(17,24,39,0.04) !important;
}

[data-testid="stFileUploader"] > div {
    background: #ffffff !important;
    border-radius: 14px !important;
    border: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f8fafc !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #111827 !important;
    fill: #111827 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #0f172a 0%, #1f3b63 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.55rem 1rem !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: white !important;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] label {
    color: #111827 !important;
}

/* ---------- METRICS ---------- */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d8e0ea;
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 8px 18px rgba(17,24,39,0.05);
}

[data-testid="stMetric"] * {
    color: #111827 !important;
}

/* ---------- TABLES ---------- */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #d8e0ea;
    box-shadow: 0 8px 18px rgba(17,24,39,0.05);
}

[data-testid="stDataFrame"] * {
    color: #111827 !important;
}

/* ---------- RESULT BANNERS ---------- */
.result-positive {
    background: linear-gradient(135deg, #e8f0fb 0%, #f8fbff 100%);
    border-left: 6px solid #1f3b63;
    padding: 14px 16px;
    border-radius: 16px;
    color: #111827 !important;
    font-weight: 700;
    margin-top: 12px;
}

.result-negative {
    background: linear-gradient(135deg, #f1f5f9 0%, #ffffff 100%);
    border-left: 6px solid #111827;
    padding: 14px 16px;
    border-radius: 16px;
    color: #111827 !important;
    font-weight: 700;
    margin-top: 12px;
}

/* ---------- ALERTS ---------- */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
model = joblib.load("advanced_sentiment_model.pkl")
vectorizer = joblib.load("advanced_tfidf_vectorizer.pkl")

def safe_read_csv(file_name):
    path = Path(file_name)
    if path.exists():
        return pd.read_csv(path)
    return None

model_results_df = safe_read_csv("model_results.csv")
detailed_results_df = safe_read_csv("detailed_model_results.csv")
error_df = safe_read_csv("model_errors.csv")
top_positive_df = safe_read_csv("top_positive_keywords.csv")
top_negative_df = safe_read_csv("top_negative_keywords.csv")
feature_results_df = safe_read_csv("feature_enhanced_results.csv")

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
NEGATIVE_HINT_WORDS = {
    "kötü", "berbat", "rezalet", "iğrenç", "nefret", "saçma",
    "boktan", "aptal", "salak", "çirkin", "sinir", "üzgün",
    "gerzek", "embesil", "amk", "orospu", "geber", "öl", "öldürürüm",
    "haketmiyorsun", "pezevenk", "sokayım", "götlek", "düşman"
}

POSITIVE_HINT_WORDS = {
    "güzel", "harika", "mükemmel", "mutlu", "seviyorum", "başarılı",
    "tebrikler", "hayırlı", "iyi", "süper", "güzelsin", "mutluyum",
    "tatlısın", "teşekkürler", "başardım", "sevgi", "düğün"
}

NEGATION_WORDS = {"değil", "yok", "hiç", "asla", "degil"}
INTENSIFIERS = {"çok", "cok", "aşırı", "asiri", "gerçekten", "gercekten", "fazla"}
STRONG_NEGATIVE_PHRASES = {"çok kötü", "cok kötü", "çok berbat", "çok rezalet", "nefret ediyorum"}
STRONG_POSITIVE_PHRASES = {"çok iyi", "cok iyi", "çok güzel", "çok mutlu", "harika", "mükemmel"}

EXAMPLE_COMMENTS = [
    "Bugün gerçekten çok mutlu oldum",
    "Bu tam bir rezalet",
    "kötüüüü",
    "Harika bir gün geçirdim",
    "Mutlu olmayı haketmiyorsun",
    "Bu kadar kötü bir şey görmedim",
    "iyi",
    "çok iyi",
    "çok kötü"
]

if "history" not in st.session_state:
    st.session_state.history = []

if "selected_example" not in st.session_state:
    st.session_state.selected_example = ""

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## Control Panel")
    st.markdown("### Final Model")
    st.write("• Advanced Logistic Regression")
    st.write("• TF-IDF n-grams")
    st.write("• Hybrid sentiment correction")

    st.markdown("### Final Metrics")
    st.write("• Accuracy: 87.19%")
    st.write("• Precision: 85.64%")
    st.write("• Recall: 92.15%")
    st.write("• F1 Score: 88.78%")

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.success("History cleared.")

# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------
def normalize_repeated_characters(text):
    return re.sub(r"(.)\1{2,}", r"\1", text)

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

    replacements = {
        "kotu": "kötü",
        "iyiim": "iyi",
        "iyii": "iyi",
        "guzel": "güzel",
        "cok": "çok",
        "berbaat": "berbat",
        "rezalett": "rezalet"
    }

    words = text.split()
    words = [replacements.get(w, w) for w in words]
    return " ".join(words)

def extract_text_diagnostics(original_text, cleaned_text):
    words = cleaned_text.split()
    return {
        "text_length": len(original_text),
        "word_count": len(words),
        "exclamation_count": original_text.count("!"),
        "question_count": original_text.count("?"),
        "uppercase_ratio": round(sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1), 4),
        "repeated_char_patterns": len(re.findall(r"(.)\1{2,}", original_text.lower())),
        "negative_word_count": sum(1 for w in words if w in NEGATIVE_HINT_WORDS),
        "positive_word_count": sum(1 for w in words if w in POSITIVE_HINT_WORDS),
    }

# --------------------------------------------------
# HYBRID ENGINE
# --------------------------------------------------
def lexicon_score(cleaned_text):
    words = cleaned_text.split()
    pos_words = [w for w in words if w in POSITIVE_HINT_WORDS]
    neg_words = [w for w in words if w in NEGATIVE_HINT_WORDS]
    score = len(pos_words) - len(neg_words)
    return score, pos_words, neg_words

def hybrid_prediction(cleaned_text):
    input_vector = vectorizer.transform([cleaned_text])
    ml_pred = model.predict(input_vector)[0]
    ml_prob = model.predict_proba(input_vector)[0]

    score, pos_words, neg_words = lexicon_score(cleaned_text)

    neg_prob, pos_prob = ml_prob[0], ml_prob[1]
    words = cleaned_text.split()
    word_set = set(words)
    triggered_rules = []

    strong_negative_words = {
        "kötü", "berbat", "rezalet", "iğrenç", "nefret",
        "aptal", "salak", "gerzek", "embesil", "amk",
        "orospu", "geber", "öldürürüm", "haketmiyorsun"
    }

    strong_positive_words = {
        "harika", "mükemmel", "mutlu", "mutluyum",
        "güzel", "güzelsin", "başarılı", "tebrikler",
        "hayırlı", "süper", "iyi"
    }

    if score >= 2:
        pos_prob = min(pos_prob + 0.15, 0.99)
        neg_prob = 1 - pos_prob
        triggered_rules.append("Positive lexicon boost")
    elif score <= -2:
        neg_prob = min(neg_prob + 0.15, 0.99)
        pos_prob = 1 - neg_prob
        triggered_rules.append("Negative lexicon boost")

    if len(words) == 1:
        if words[0] in strong_negative_words:
            neg_prob = max(neg_prob, 0.92)
            pos_prob = 1 - neg_prob
            triggered_rules.append("Single strong negative word rule")
        if words[0] in strong_positive_words:
            pos_prob = max(pos_prob, 0.92)
            neg_prob = 1 - pos_prob
            triggered_rules.append("Single strong positive word rule")

    if len(word_set & strong_negative_words) >= 1 and pos_prob > 0.70:
        neg_prob = max(neg_prob, 0.78)
        pos_prob = 1 - neg_prob
        triggered_rules.append("Strong negative word override")

    if len(word_set & strong_positive_words) >= 1 and neg_prob > 0.70:
        pos_prob = max(pos_prob, 0.78)
        neg_prob = 1 - pos_prob
        triggered_rules.append("Strong positive word override")

    if cleaned_text in STRONG_NEGATIVE_PHRASES or "çok kötü" in cleaned_text or "çok berbat" in cleaned_text:
        neg_prob = max(neg_prob, 0.93)
        pos_prob = 1 - neg_prob
        triggered_rules.append("Strong negative phrase rule")

    if cleaned_text in STRONG_POSITIVE_PHRASES or "çok iyi" in cleaned_text or "çok güzel" in cleaned_text:
        pos_prob = max(pos_prob, 0.93)
        neg_prob = 1 - pos_prob
        triggered_rules.append("Strong positive phrase rule")

    if any(w in INTENSIFIERS for w in words):
        if any(w in strong_negative_words for w in words):
            neg_prob = max(neg_prob, 0.90)
            pos_prob = 1 - neg_prob
            triggered_rules.append("Negative intensifier rule")
        if any(w in strong_positive_words for w in words):
            pos_prob = max(pos_prob, 0.90)
            neg_prob = 1 - pos_prob
            triggered_rules.append("Positive intensifier rule")

    if "not_mutlu" in cleaned_text or "not_iyi" in cleaned_text:
        neg_prob = max(neg_prob, 0.88)
        pos_prob = 1 - neg_prob
        triggered_rules.append("Negation rule")

    if "haketmiyorsun" in cleaned_text or "nefret" in cleaned_text:
        neg_prob = max(neg_prob, 0.88)
        pos_prob = 1 - neg_prob
        triggered_rules.append("Error-driven rule")

    final_pred = 1 if pos_prob >= neg_prob else 0
    confidence = max(pos_prob, neg_prob)

    if final_pred == 0:
        sentiment_strength = "Very Negative" if confidence >= 0.90 else "Negative" if confidence >= 0.75 else "Slightly Negative"
    else:
        sentiment_strength = "Very Positive" if confidence >= 0.90 else "Positive" if confidence >= 0.75 else "Slightly Positive"

    confidence_label = "High Confidence" if confidence >= 0.90 else "Medium Confidence" if confidence >= 0.75 else "Low Confidence"

    return {
        "ml_prediction": ml_pred,
        "final_prediction": final_pred,
        "negative_probability": neg_prob,
        "positive_probability": pos_prob,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "sentiment_strength": sentiment_strength,
        "lexicon_score": score,
        "positive_words_found": pos_words,
        "negative_words_found": neg_words,
        "triggered_rules": triggered_rules
    }

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def create_batch_summary_chart(df):
    counts = df["prediction"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(counts.index, counts.values)
    ax.set_title("Batch Prediction Summary")
    ax.set_ylabel("Count")
    return fig

def detect_text_column(df):
    candidates = ["text", "comment", "yorum", "Paylaşım", "paylaşım", "message"]
    for col in candidates:
        if col in df.columns:
            return col
    return None

# --------------------------------------------------
# TOP HERO
# --------------------------------------------------
col_logo, col_title = st.columns([1, 5])

with col_logo:
    if Path("mef_logo.png").exists():
        st.image("mef_logo.png", width=120)

with col_title:
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">MEF Sentiment Analysis Dashboard</div>
        <div class="hero-subtitle">Readable, elegant, and presentation-ready explainable sentiment analysis system for Turkish social media comments.</div>
    </div>
    """, unsafe_allow_html=True)

# TOP SUMMARY CARDS
summary1, summary2, summary3 = st.columns(3)

with summary1:
    st.markdown("""
    <div class="top-card">
        <div class="top-card-title">Final Model</div>
        <div class="top-card-text">Advanced Logistic Regression with TF-IDF n-grams and hybrid sentiment correction.</div>
    </div>
    """, unsafe_allow_html=True)

with summary2:
    st.markdown("""
    <div class="top-card">
        <div class="top-card-title">Best Performance</div>
        <div class="top-card-text">Accuracy: 87.19%<br>Precision: 85.64%<br>Recall: 92.15%<br>F1 Score: 88.78%</div>
    </div>
    """, unsafe_allow_html=True)

with summary3:
    st.markdown("""
    <div class="top-card">
        <div class="top-card-title">Project Strength</div>
        <div class="top-card-text">Explainable hybrid model, batch prediction, keyword insights, error analysis, and interactive dashboard.</div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Prediction",
    "Batch Prediction",
    "Project Insights",
    "Prediction History"
])

# --------------------------------------------------
# TAB 1
# --------------------------------------------------
with tab1:
    st.markdown('<div class="panel-card"><div class="panel-title">Single Comment Prediction</div>', unsafe_allow_html=True)

    st.write("### Quick Examples")
    example_cols = st.columns(3)
    for i, comment in enumerate(EXAMPLE_COMMENTS):
        if example_cols[i % 3].button(comment, key=f"example_{i}"):
            st.session_state.selected_example = comment

    st.write("### Built-in Test Set")
    if st.button("Run Quick Test Suite"):
        test_comments = [
            "kötüüüü", "çok kötü", "iyi", "çok iyi",
            "harika", "berbat", "Mutlu olmayı haketmiyorsun",
            "Bugün gerçekten çok mutlu oldum"
        ]
        test_results = []
        for text in test_comments:
            cleaned = clean_text(text)
            result = hybrid_prediction(cleaned)
            test_results.append({
                "input_text": text,
                "cleaned_text": cleaned,
                "prediction": "Positive" if result["final_prediction"] == 1 else "Negative",
                "sentiment_strength": result["sentiment_strength"],
                "confidence": round(result["confidence"], 4),
                "negative_probability": round(result["negative_probability"], 4),
                "positive_probability": round(result["positive_probability"], 4)
            })
        st.dataframe(pd.DataFrame(test_results), width="stretch")

    user_input = st.text_area(
        "Enter your comment here:",
        value=st.session_state.selected_example,
        height=120
    )

    col_a, col_b = st.columns([1, 1])
    predict_clicked = col_a.button("Predict Sentiment")
    clear_clicked = col_b.button("Clear Input")

    if clear_clicked:
        st.session_state.selected_example = ""
        st.rerun()

    if predict_clicked:
        if user_input.strip():
            cleaned_input = clean_text(user_input)
            result = hybrid_prediction(cleaned_input)
            diagnostics = extract_text_diagnostics(user_input, cleaned_input)

            ml_label = "Positive" if result["ml_prediction"] == 1 else "Negative"
            final_label = "Positive" if result["final_prediction"] == 1 else "Negative"

            st.write("## Prediction Result")

            result_cols = st.columns(5)
            result_cols[0].metric("Final Prediction", final_label)
            result_cols[1].metric("Sentiment Strength", result["sentiment_strength"])
            result_cols[2].metric("Confidence", f"{result['confidence']:.4f}")
            result_cols[3].metric("Negative Prob.", f"{result['negative_probability']:.4f}")
            result_cols[4].metric("Positive Prob.", f"{result['positive_probability']:.4f}")

            st.write("### Prediction Explanation")
            explain_col1, explain_col2 = st.columns(2)

            with explain_col1:
                st.write(f"**Original ML Prediction:** {ml_label}")
                st.write(f"**Cleaned Text:** {cleaned_input}")
                st.write(f"**Lexicon Score:** {result['lexicon_score']}")
                st.write(f"**Confidence Level:** {result['confidence_label']}")

            with explain_col2:
                st.write(f"**Positive Words Found:** {', '.join(result['positive_words_found']) if result['positive_words_found'] else 'None'}")
                st.write(f"**Negative Words Found:** {', '.join(result['negative_words_found']) if result['negative_words_found'] else 'None'}")
                st.write(f"**Triggered Rules:** {', '.join(result['triggered_rules']) if result['triggered_rules'] else 'None'}")

            st.write("### Text Diagnostics")
            diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
            diag_col1.metric("Text Length", diagnostics["text_length"])
            diag_col2.metric("Word Count", diagnostics["word_count"])
            diag_col3.metric("Exclamation Count", diagnostics["exclamation_count"])
            diag_col4.metric("Question Count", diagnostics["question_count"])

            diag_col5, diag_col6, diag_col7, diag_col8 = st.columns(4)
            diag_col5.metric("Uppercase Ratio", diagnostics["uppercase_ratio"])
            diag_col6.metric("Repeated Patterns", diagnostics["repeated_char_patterns"])
            diag_col7.metric("Positive Word Count", diagnostics["positive_word_count"])
            diag_col8.metric("Negative Word Count", diagnostics["negative_word_count"])

            if result["final_prediction"] == 1:
                st.markdown(f'<div class="result-positive">The final system predicts that this comment is positive ({result["sentiment_strength"]}).</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-negative">The final system predicts that this comment is negative ({result["sentiment_strength"]}).</div>', unsafe_allow_html=True)

            if result["confidence"] < 0.65:
                st.warning("Low-confidence prediction. This comment may be ambiguous, sarcastic, or highly informal.")
            elif result["confidence"] < 0.90:
                st.info("Prediction confidence is moderate. The result looks meaningful but not absolute.")
            else:
                st.info("Prediction confidence is high.")

            single_result_df = pd.DataFrame([{
                "original_text": user_input,
                "cleaned_text": cleaned_input,
                "ml_prediction": ml_label,
                "final_prediction": final_label,
                "sentiment_strength": result["sentiment_strength"],
                "confidence": round(result["confidence"], 4),
                "negative_probability": round(result["negative_probability"], 4),
                "positive_probability": round(result["positive_probability"], 4),
                "lexicon_score": result["lexicon_score"],
                "positive_words_found": ", ".join(result["positive_words_found"]) if result["positive_words_found"] else "",
                "negative_words_found": ", ".join(result["negative_words_found"]) if result["negative_words_found"] else "",
                "triggered_rules": ", ".join(result["triggered_rules"]) if result["triggered_rules"] else ""
            }])

            single_buffer = BytesIO()
            single_result_df.to_csv(single_buffer, index=False, encoding="utf-8-sig")
            st.download_button(
                label="Download This Prediction",
                data=single_buffer.getvalue(),
                file_name="single_prediction_result.csv",
                mime="text/csv"
            )

            st.session_state.history.append({
                "original_text": user_input,
                "cleaned_text": cleaned_input,
                "ml_prediction": ml_label,
                "final_prediction": final_label,
                "negative_probability": round(result["negative_probability"], 4),
                "positive_probability": round(result["positive_probability"], 4),
                "confidence": round(result["confidence"], 4)
            })
        else:
            st.warning("Please enter a comment.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# TAB 2
# --------------------------------------------------
with tab2:
    st.markdown('<div class="panel-card"><div class="panel-title">Batch CSV Prediction</div>', unsafe_allow_html=True)
    st.write("Upload a CSV file. The app will automatically detect the text column if possible.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        detected_col = detect_text_column(batch_df)

        if detected_col is not None:
            st.success(f"Detected text column: {detected_col}")

            batch_df["clean_text"] = batch_df[detected_col].astype(str).apply(clean_text)

            results = []
            for txt in batch_df["clean_text"]:
                result = hybrid_prediction(txt)
                results.append({
                    "prediction": "Positive" if result["final_prediction"] == 1 else "Negative",
                    "negative_probability": round(result["negative_probability"], 4),
                    "positive_probability": round(result["positive_probability"], 4),
                    "confidence": round(result["confidence"], 4)
                })

            results_df = pd.DataFrame(results)
            final_df = pd.concat([batch_df, results_df], axis=1)

            st.dataframe(final_df, width="stretch")

            positive_count = (final_df["prediction"] == "Positive").sum()
            negative_count = (final_df["prediction"] == "Negative").sum()
            avg_confidence = final_df["confidence"].mean()
            total_rows = len(final_df)

            summary_cols = st.columns(4)
            summary_cols[0].metric("Total Comments", int(total_rows))
            summary_cols[1].metric("Positive", int(positive_count))
            summary_cols[2].metric("Negative", int(negative_count))
            summary_cols[3].metric("Avg. Confidence", f"{avg_confidence:.4f}")

            fig = create_batch_summary_chart(final_df)
            st.pyplot(fig)

            csv_buffer = BytesIO()
            final_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="batch_sentiment_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("No suitable text column found. Expected columns like: text, comment, yorum, Paylaşım.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# TAB 3
# --------------------------------------------------
with tab3:
    st.markdown('<div class="panel-card"><div class="panel-title">Project Insights</div>', unsafe_allow_html=True)

    insight_cols = st.columns(4)
    insight_cols[0].metric("Final Accuracy", "87.19%")
    insight_cols[1].metric("Final Precision", "85.64%")
    insight_cols[2].metric("Final Recall", "92.15%")
    insight_cols[3].metric("Final F1 Score", "88.78%")

    if detailed_results_df is not None:
        st.write("### Detailed Model Results")
        st.dataframe(detailed_results_df, width="stretch")
    elif model_results_df is not None:
        st.write("### Model Results")
        st.dataframe(model_results_df, width="stretch")

    if feature_results_df is not None:
        st.write("### Experimental Feature-Enhanced Model")
        st.dataframe(feature_results_df, width="stretch")
        st.info("This experimental model combined TF-IDF with structural text features, but the advanced hybrid model remained the best-performing final system.")

    if top_positive_df is not None:
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.write("### Top Positive Keywords")
            st.dataframe(top_positive_df.head(10), width="stretch")
        if top_negative_df is not None:
            with col_neg:
                st.write("### Top Negative Keywords")
                st.dataframe(top_negative_df.head(10), width="stretch")

    st.write("### Insights")
    st.write("The model performs strongly overall, but still faces challenges with sarcasm, mixed sentiment, informal social media language, and context-dependent profanity usage.")

    image_files = [
        "accuracy_comparison.png",
        "f1_comparison.png",
        "class_distribution.png",
        "positive_wordcloud.png",
        "negative_wordcloud.png"
    ]

    for image_file in image_files:
        if Path(image_file).exists():
            st.image(image_file, caption=image_file, width="stretch")

    if error_df is not None:
        st.write("### Sample Error Analysis")
        st.dataframe(error_df.head(10), width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# TAB 4
# --------------------------------------------------
with tab4:
    st.markdown('<div class="panel-card"><div class="panel-title">Prediction History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, width="stretch")

        history_buffer = BytesIO()
        history_df.to_csv(history_buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download History as CSV",
            data=history_buffer.getvalue(),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions yet.")

    st.markdown("</div>", unsafe_allow_html=True)
