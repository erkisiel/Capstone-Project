import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Energy Dashboard", layout="wide")

model = joblib.load("final_energy_model.pkl")
features = joblib.load("energy_features.pkl")
df = pd.read_csv("advanced_cleaned_energy.csv")

st.markdown("""
<style>
.stApp {
    background: #f4f7fb !important;
}

/* MAIN TEXT DARK */
section.main,
section.main *,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] *,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #111827 !important;
}

/* FORCE ALL HEADINGS DARK */
section.main h1,
section.main h2,
section.main h3,
section.main h4,
section.main h5,
section.main h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {
    color: #111827 !important;
    font-weight: 800 !important;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 34px;
    border-radius: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.18);
}
.hero,
.hero *,
.hero h1,
.hero p {
    color: white !important;
}

/* CARD */
.card {
    background: #ffffff !important;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
}
.card,
.card * {
    color: #111827 !important;
}

/* TEXT BOX */
.text-box {
    background: #ffffff !important;
    color: #111827 !important;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    font-size: 18px;
    line-height: 1.7;
}
.text-box,
.text-box * {
    color: #111827 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #0f172a !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: white !important;
}

/* NUMBER INPUT */
[data-testid="stNumberInput"] > div {
    background: white !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stNumberInput"] input {
    background: white !important;
    color: #111827 !important;
}
[data-testid="stNumberInput"] button {
    background: #e5e7eb !important;
    color: #111827 !important;
}
[data-testid="stNumberInput"] button * {
    color: #111827 !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #ffffff, #f1f5f9) !important;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
}
[data-testid="stMetric"] * {
    color: #111827 !important;
}

/* TABS */
button[role="tab"] {
    color: #111827 !important;
    font-weight: 700 !important;
    background: #e8eef7 !important;
    border-radius: 10px !important;
    margin-right: 6px !important;
}
button[role="tab"] * {
    color: #111827 !important;
}
button[role="tab"][aria-selected="true"] {
    background: #0f172a !important;
}
button[role="tab"][aria-selected="true"] * {
    color: white !important;
}

/* DOWNLOAD BUTTON */
.stDownloadButton button {
    background: #0f172a !important;
    color: white !important;
    border-radius: 10px !important;
}
.stDownloadButton button * {
    color: white !important;
}

/* REMOVE WEIRD TEXT SHADOW */
* {
    text-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


def make_input():
    return pd.DataFrame([{
        "hour": hour,
        "day": day,
        "month": month,
        "year": year,
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week in [5, 6] else 0,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_24": rolling_24,
        "rolling_168": rolling_168
    }])[features]


def plot_line(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(data, linewidth=2)
    ax.set_title(title, color="#111827")
    ax.set_xlabel(xlabel, color="#111827")
    ax.set_ylabel(ylabel, color="#111827")
    ax.tick_params(colors="#111827")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


st.markdown("""
<div class="hero">
<h1>⚡ Energy Consumption Forecasting Dashboard</h1>
<p>Predict future energy usage using historical time-series data and machine learning.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Prediction Inputs")

hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.slider("Year", 2004, 2018, 2010)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 2)

st.sidebar.subheader("Historical Energy Inputs")

st.sidebar.markdown("**Last Hour**")
lag_1 = st.sidebar.number_input(" ", value=float(df["energy"].iloc[-1]), key="lag_1")

st.sidebar.markdown("**Yesterday**")
lag_24 = st.sidebar.number_input(" ", value=float(df["energy"].iloc[-24]), key="lag_24")

st.sidebar.markdown("**Last Week**")
lag_168 = st.sidebar.number_input(" ", value=float(df["energy"].iloc[-168]), key="lag_168")

st.sidebar.markdown("**24h Avg**")
rolling_24 = st.sidebar.number_input(" ", value=float(df["rolling_24"].iloc[-1]), key="rolling_24")

st.sidebar.markdown("**Weekly Avg**")
rolling_168 = st.sidebar.number_input(" ", value=float(df["rolling_168"].iloc[-1]), key="rolling_168")

input_df = make_input()
prediction = model.predict(input_df)[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Energy", f"{prediction:.2f} MW")
c2.metric("Final Test R²", "0.9910")
c3.metric("Final Test MAE", "25.37 MW")
c4.metric("Final Test RMSE", "35.69 MW")

if prediction >= 2500:
    st.error("⚠️ High energy consumption expected.")
elif prediction <= 1500:
    st.info("ℹ️ Low energy consumption period expected.")
else:
    st.success("✅ Normal energy consumption level expected.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary",
    "Prediction",
    "24-Hour Forecast",
    "Scenario Comparison",
    "Data Analysis",
    "Advanced Validation"
])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Project Objective")
    st.markdown("""
    <div class="text-box">
    The main objective of this project is to forecast hourly energy consumption using historical time-series data.
    The system uses previous energy values, rolling averages, and calendar-based features to predict future demand.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Why This Project Matters")
    st.markdown("""
    <div class="text-box">
    Energy forecasting helps support demand planning, peak-hour detection, and resource management.
    This dashboard works as a decision-support system for estimating future energy usage.
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("Dataset Rows", f"{df.shape[0]:,}")
    b.metric("Missing Values", int(df.isnull().sum().sum()))
    c.metric("Final Model", "Tuned Decision Tree")
    d.metric("Test R²", "0.9910")

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Single Prediction")
    st.dataframe(input_df, width="stretch")
    st.success(f"Predicted Energy Consumption: {prediction:.2f} MW")

    feature_explanation = pd.DataFrame({
        "Feature": [
            "hour", "day", "month", "year", "day_of_week", "is_weekend",
            "lag_1", "lag_24", "lag_168", "rolling_24", "rolling_168"
        ],
        "Meaning": [
            "Hour of the day",
            "Day of the month",
            "Month of the year",
            "Year",
            "Day of week",
            "Weekend indicator",
            "Previous hour energy",
            "Same hour yesterday",
            "Same hour last week",
            "Last 24-hour average",
            "Last weekly average"
        ]
    })

    st.subheader("Feature Explanation")
    st.dataframe(feature_explanation, width="stretch")

    st.subheader("Recent Energy Trend")
    plot_line(df["energy"].iloc[-500:].values, "Recent Energy Trend", "Time Index", "Energy (MW)")

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Next 24-Hour Forecast")

    preds = []
    temp_lag = lag_1

    for i in range(24):
        temp = input_df.copy()
        temp["hour"] = (hour + i) % 24
        temp["lag_1"] = temp_lag
        p = model.predict(temp)[0]
        preds.append(p)
        temp_lag = p

    forecast_df = pd.DataFrame({
        "Hour Ahead": range(1, 25),
        "Predicted Energy": preds
    })

    plot_line(forecast_df["Predicted Energy"].values, "Next 24-Hour Forecast", "Hour Ahead", "Predicted Energy (MW)")
    st.dataframe(forecast_df, width="stretch")

    st.download_button(
        "Download 24-Hour Forecast",
        forecast_df.to_csv(index=False).encode("utf-8"),
        "next_24_hour_forecast.csv",
        "text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Scenario Comparison")

    h2 = st.slider("Second Scenario Hour", 0, 23, 18)

    scenario_b = input_df.copy()
    scenario_b["hour"] = h2

    prediction_b = model.predict(scenario_b)[0]

    s1, s2, s3 = st.columns(3)
    s1.metric("Scenario A", f"{prediction:.2f} MW")
    s2.metric("Scenario B", f"{prediction_b:.2f} MW")
    s3.metric("Difference", f"{abs(prediction_b - prediction):.2f} MW")

    comparison_df = pd.DataFrame({
        "Scenario": ["Scenario A", "Scenario B"],
        "Predicted Energy": [prediction, prediction_b]
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(comparison_df["Scenario"], comparison_df["Predicted Energy"])
    ax.set_title("Scenario Comparison", color="#111827")
    ax.set_ylabel("Predicted Energy (MW)", color="#111827")
    ax.tick_params(colors="#111827")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Dataset Overview")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{df.shape[0]:,}")
    d2.metric("Columns", df.shape[1])
    d3.metric("Missing Values", int(df.isnull().sum().sum()))
    d4.metric("Average Energy", f"{df['energy'].mean():.2f} MW")

    st.subheader("First Rows")
    st.dataframe(df.head(), width="stretch")

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), width="stretch")

    st.subheader("Visual Outputs")

    visuals = [
        "energy_distribution_full.png",
        "energy_boxplot.png",
        "avg_energy_by_hour_full.png",
        "avg_energy_by_month_full.png",
        "correlation_matrix_energy.png",
        "energy_time_series_sample.png",
        "final_feature_importance.png",
        "final_error_distribution.png",
        "final_actual_vs_predicted_line.png",
        "baseline_vs_final_mae.png",
        "error_by_hour_final.png",
        "error_by_month_final.png"
    ]

    for file in visuals:
        if Path(file).exists():
            st.image(file, caption=file, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Advanced Validation and Explainability")

    st.markdown("""
    <div class="text-box">
    This section includes TimeSeriesSplit validation, residual analysis, autocorrelation analysis, and SHAP explainability.
    These outputs show that the model is stable, interpretable, and suitable for time-series forecasting.
    </div>
    """, unsafe_allow_html=True)

    validation_files = [
        ("time_series_cv_r2.png", "Time-Series Cross Validation R² Scores"),
        ("residual_distribution.png", "Residual Distribution"),
        ("residual_vs_predicted.png", "Residuals vs Predicted Values"),
        ("autocorrelation_energy.png", "Autocorrelation Analysis"),
        ("shap_summary_plot.png", "SHAP Summary Plot")
    ]

    for file, caption in validation_files:
        if Path(file).exists():
            st.image(file, caption=caption, width="stretch")

    if Path("advanced_validation_summary.csv").exists():
        st.subheader("Advanced Validation Summary")
        validation_summary = pd.read_csv("advanced_validation_summary.csv")
        st.dataframe(validation_summary, width="stretch")

    if Path("time_series_cv_results.csv").exists():
        st.subheader("Time-Series Cross Validation Results")
        cv_results = pd.read_csv("time_series_cv_results.csv")
        st.dataframe(cv_results, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)
