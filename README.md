# AI & Data Science Capstone Projects

This repository presents two capstone projects completed as part of my **Artificial Intelligence & Data Science minor at MEF University**.

The projects focus on two different areas of applied machine learning:

1. **Energy Consumption Forecasting** — time-series regression and explainable machine learning  
2. **Turkish Social Media Sentiment Analysis** — natural language processing and binary text classification

Both projects cover the complete machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model comparison, evaluation, interpretation, and interactive deployment with Streamlit.

---

## Projects

### 1. Energy Consumption Forecasting Using Machine Learning

A machine learning pipeline for predicting **hourly electricity consumption** from historical energy demand data.

The dataset contains **121,107 hourly observations spanning 2004–2018**. Temporal, lag-based, and rolling-window features were engineered from the original time-series data.

#### Models
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

#### Key techniques
- Time-series preprocessing
- Temporal feature engineering
- Lag features: `lag_1`, `lag_24`, `lag_168`
- Rolling statistics
- Chronological train/test split
- GridSearchCV
- Time-series cross-validation
- SHAP explainability
- Streamlit deployment

#### Results
The tuned Decision Tree achieved approximately:

| Metric | Result |
|---|---:|
| MAE | ~26 MW |
| RMSE | ~51 MW |
| Test R² | ~0.9999 |
| 5-fold CV R² | 0.9888 |

SHAP analysis showed that the **1-hour lag feature (`lag_1`) contributed approximately 95% of total feature importance**, highlighting the strong short-term autocorrelation in electricity demand.

The project was also deployed as an interactive **Streamlit forecasting dashboard** with exploratory analysis, model comparison, live prediction, and prediction explanations.

**Full report:** [Energy Consumption Forecasting Report](reports/energy_consumption_forecasting_report.pdf)

---

### 2. Turkish Social Media Sentiment Analysis

A Natural Language Processing project for classifying Turkish social media comments as **Positive** or **Negative**.

The final dataset contains **11,117 labeled Turkish social media comments**. The project investigates how classical machine learning methods perform on Turkish text using TF-IDF representations.

#### NLP pipeline
- Text cleaning and normalization
- URL removal
- Mention and hashtag removal
- Turkish-character-aware preprocessing
- Stop-word processing
- TF-IDF vectorization
- Unigrams and bigrams
- Stratified train/test split

#### Models
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

#### Results

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Logistic Regression | ~86.2% | ~0.88 |
| Naive Bayes | ~85.8% | ~0.87 |
| Linear SVM | ~85.9% | ~0.87 |

The analysis also identified an important Turkish NLP challenge: **negation and agglutinative morphology** can cause bag-of-words models to misinterpret phrases whose sentiment depends on context.

The project includes an interactive **Streamlit application** for real-time sentiment prediction and model inspection.

**Full report:** [Turkish Social Media Sentiment Analysis Report](reports/turkish_sentiment_analysis_report.pdf)

---

## Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **SHAP**
- **TF-IDF**
- Machine Learning
- Natural Language Processing
- Time-Series Forecasting
- Data Visualization
- Model Evaluation

---

## Machine Learning Workflow

Both projects were developed using an end-to-end applied machine learning workflow:

```text
Data Collection / Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Comparison
        ↓
Performance Evaluation
        ↓
Explainability / Error Analysis
        ↓
Interactive Streamlit Application
```

---

## Repository Structure

```text
Capstone-Project/
│
├── energy-consumption-forecasting/
│   └── project files
│
├── turkish-sentiment-analysis/
│   └── project files
│
├── reports/
│   ├── energy_consumption_forecasting_report.pdf
│   └── turkish_sentiment_analysis_report.pdf
│
└── README.md
```

> The exact source-code folders can be adjusted to match the current repository structure.

---

## Academic Context

These projects were completed for **DSAI 302 – Capstone Project** at **MEF University** as part of my **Artificial Intelligence & Data Science minor**.

**Student:** Elif Erkişi  
**Instructor:** Tuna Çakar  
**Year:** 2026

---

## What I Practiced

Through these projects, I worked on:

- Building complete machine learning pipelines
- Working with structured and unstructured datasets
- Time-series feature engineering
- Turkish NLP preprocessing
- Regression and classification
- Hyperparameter tuning
- Cross-validation
- Model interpretation with SHAP
- Error and misclassification analysis
- Interactive ML application development with Streamlit

---

## Future Improvements

### Energy Forecasting
- Add meteorological variables such as temperature and humidity
- Explore multi-step forecasting
- Experiment with LSTM and Transformer-based forecasting models
- Evaluate generalization across different geographic regions

### Turkish Sentiment Analysis
- Fine-tune Turkish transformer models such as BERTurk
- Improve modeling of Turkish negation and morphology
- Add subword tokenization
- Extend the dataset
- Explore domain-specific sentiment features

---

## Reports

- [Energy Consumption Forecasting – Full Report](reports/energy_consumption_forecasting_report.pdf)
- [Turkish Social Media Sentiment Analysis – Full Report](reports/turkish_sentiment_analysis_report.pdf)

---

## Author

**Elif Erkişi**  
Computer Engineering  
Artificial Intelligence & Data Science Minor — Completed  
MEF University
