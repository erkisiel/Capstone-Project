import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# LOAD FILES
# ==================================================
model = joblib.load("final_energy_model.pkl")
features = joblib.load("energy_features.pkl")

df = pd.read_csv("advanced_cleaned_energy.csv")
results_df = pd.read_csv("advanced_energy_model_results.csv") if Path("advanced_energy_model_results.csv").exists() else None
pred_df = pd.read_csv("final_energy_predictions.csv") if Path("final_energy_predictions.csv").exists() else None
importance_df = pd.read_csv("final_feature_importance.csv") if Path("final_feature_importance.csv").exists() else None

if pred_df is not None and "absolute_error" not in pred_df.columns:
    pred_df["absolute_error"] = abs(pred_df["actual_energy"] - pred_df["predicted_energy"])

# ==================================================
# CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f4f7fb 0%, #eef2f7 100%);
}

/* MAIN TEXT FIX */
.main, .main p, .main div, .main span, .main label,
.stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #111827 !important;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #111827 100%);
    padding: 30px;
    border-radius: 26px;
    color: white !important;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.22);
    margin-bottom: 20px;
}

.hero h1 {
    color: white !important;
    margin-bottom: 8px;
    font-size: 2.35rem;
}

.hero p {
    color: #e5e7eb !important;
    font-size: 1rem;
    margin-bottom: 0;
}

/* CARDS */
.card {
    background: white;
    border: 1px solid #d8e0ea;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
    margin-bottom: 18px;
}

.card, .card p, .card div, .card span, .card label,
.info-box, .info-box p, .info-box div, .info-box span,
.warning-box, .warning-box p, .warning-box div, .warning-box span,
.success-box, .success-box p, .success-box div, .success-box span {
    color: #111827 !important;
}

.info-box {
    background: #f8fafc;
    border-left: 6px solid #1e3a5f;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
}

.warning-box {
    background: #fff7ed;
    border-left: 6px solid #f97316;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
}

.success-box {
    background: #ecfdf5;
    border-left: 6px solid #10b981;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: white !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d8e0ea;
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetric"] * {
    color: #111827 !important;
}

/* BUTTONS */
.stButton button {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    color: white !important;
    border-radius: 12px;
    border: none;
    font-weight: 700;
}

.stButton button * {
    color: white !important;
}

.stDownloadButton button {
    background: linear-gradient(135deg, #111827 0%, #1e3a5f 100%);
    color: white !important;
    border-radius: 12px;
    border: none;
    font-weight: 700;
}

.stDownloadButton button * {
    color: white !important;
}

/* TABS */
button[role="tab"] {
    color: #111827 !important;
    font-weight: 700 !important;
}

button[role="tab"][aria-selected="true"] {
    background: #0f172a !important;
    color: white !important;
    border-radius: 12px !important;
}

button[role="tab"][aria-selected="true"] * {
    color: white !important;
}

/* DATAFRAME TEXT */
[data-testid="stDataFrame"] * {
    color: #111827 !important;
}

/* INPUTS */
input, textarea {
    color: #111827 !important;
    background: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def make_input_df(hour, day, month, year, day_of_week, lag_1, lag_24, lag_168, rolling_24, rolling_168):
    is_weekend = 1 if day_of_week in [5, 6] else 0

    input_data = pd.DataFrame([{
        "hour": hour,
        "day": day,
        "month": month,
        "year": year,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_24": rolling_24,
        "rolling_168": rolling_168
    }])

    return input_data[features]


def predict_energy(input_data):
    return model.predict(input_data)[0]


def create_next_24_forecast(base_input):
    forecasts = []
    current_input = base_input.copy()

    current_lag_1 = float(current_input["lag_1"].iloc[0])
    current_lag_24 = float(current_input["lag_24"].iloc[0])
    current_lag_168 = float(current_input["lag_168"].iloc[0])
    current_rolling_24 = float(current_input["rolling_24"].iloc[0])
    current_rolling_168 = float(current_input["rolling_168"].iloc[0])

    base_hour = int(current_input["hour"].iloc[0])
    base_day = int(current_input["day"].iloc[0])
    base_month = int(current_input["month"].iloc[0])
    base_year = int(current_input["year"].iloc[0])
    base_dow = int(current_input["day_of_week"].iloc[0])

    for step in range(1, 25):
        next_hour = (base_hour + step) % 24
        next_day = base_day
        next_month = base_month
        next_year = base_year
        next_dow = (base_dow + ((base_hour + step) // 24)) % 7
        next_is_weekend = 1 if next_dow in [5, 6] else 0

        forecast_input = pd.DataFrame([{
            "hour": next_hour,
            "day": next_day,
            "month": next_month,
            "year": next_year,
            "day_of_week": next_dow,
            "is_weekend": next_is_weekend,
            "lag_1": current_lag_1,
            "lag_24": current_lag_24,
            "lag_168": current_lag_168,
            "rolling_24": current_rolling_24,
            "rolling_168": current_rolling_168
        }])[features]

        pred = float(model.predict(forecast_input)[0])

        forecasts.append({
            "forecast_hour_ahead": step,
            "hour": next_hour,
            "predicted_energy": pred
        })

        current_lag_1 = pred
        current_rolling_24 = (current_rolling_24 * 23 + pred) / 24
        current_rolling_168 = (current_rolling_168 * 167 + pred) / 168

    return pd.DataFrame(forecasts)


# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <h1>⚡ Energy Consumption Forecasting Dashboard</h1>
    <p>Advanced time-series machine learning dashboard for hourly energy consumption prediction, scenario comparison, peak analysis, and 24-hour forecasting.</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR INPUTS
# ==================================================
st.sidebar.header("Prediction Inputs")

hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.slider("Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 2)

st.sidebar.subheader("Historical Energy Inputs")

lag_1 = st.sidebar.number_input("Last Hour Energy", value=float(df["energy"].iloc[-1]))
lag_24 = st.sidebar.number_input("Same Hour Yesterday", value=float(df["energy"].iloc[-24]))
lag_168 = st.sidebar.number_input("Same Hour Last Week", value=float(df["energy"].iloc[-168]))
rolling_24 = st.sidebar.number_input("24-Hour Rolling Average", value=float(df["rolling_24"].iloc[-1]))
rolling_168 = st.sidebar.number_input("Weekly Rolling Average", value=float(df["rolling_168"].iloc[-1]))

input_df = make_input_df(
    hour, day, month, year, day_of_week,
    lag_1, lag_24, lag_168, rolling_24, rolling_168
)

prediction = predict_energy(input_df)
trend_delta = prediction - lag_1

# ==================================================
# TOP METRICS
# ==================================================
m1, m2, m3, m4 = st.columns(4)

m1.metric("Predicted Energy", f"{prediction:.2f} MW", delta=f"{trend_delta:.2f} MW")
m2.metric("Final Test R²", "0.9910")
m3.metric("Final Test MAE", "25.37 MW")
m4.metric("Final Test RMSE", "35.69 MW")

if prediction >= 2500:
    st.error("⚠️ High energy consumption expected. This may indicate a peak demand period.")
elif prediction <= 1500:
    st.info("ℹ️ Low energy consumption period expected.")
else:
    st.success("✅ Normal energy consumption level expected.")

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Executive Summary",
    "Prediction",
    "24-Hour Forecast",
    "Scenario Comparison",
    "Data Analysis",
    "Model Results",
    "Error & Peak Analysis",
    "Downloads"
])

# ==================================================
# TAB 1
# ==================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Project Objective")

    st.markdown("""
    <div class="info-box">
    The main objective of this project is to forecast hourly energy consumption using historical time-series data.
    The dashboard transforms raw date-time and energy records into machine-learning-ready features and predicts future demand.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{df.shape[0]:,}")
    c2.metric("Missing Values", int(df.isnull().sum().sum()))
    c3.metric("Final Model", "Tuned Decision Tree")
    c4.metric("Test R²", "0.9910")

    st.write("### What makes this project advanced?")
    st.write("""
    - Time-series feature engineering was applied.
    - Lag features capture previous energy behavior.
    - Rolling averages capture smoothed demand trends.
    - Multiple machine learning models were compared.
    - Final model was tuned to reduce overfitting.
    - Dashboard includes forecasting, scenario comparison, error analysis, and downloadable outputs.
    """)

    st.write("### Business Value")
    st.write("""
    Energy forecasting can support demand planning, load management, peak-hour detection, and operational decision making.
    A reliable model can help decision makers estimate future demand and prepare resources more effectively.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 2
# ==================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Single Prediction")

    st.write("""
    This prediction uses calendar features, lag features, and rolling averages.
    The model estimates expected hourly energy consumption based on current time information and historical demand.
    """)

    st.dataframe(input_df, width="stretch")
    st.success(f"Predicted Energy Consumption: {prediction:.2f} MW")

    st.write("### Feature Explanation")
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
            "Day of week from 0 to 6",
            "Whether the day is weekend",
            "Energy consumption in the previous hour",
            "Energy consumption at the same hour yesterday",
            "Energy consumption at the same hour last week",
            "Average energy consumption over last 24 hours",
            "Average energy consumption over last 168 hours"
        ]
    })
    st.dataframe(feature_explanation, width="stretch")

    st.write("### Recent Energy Trend")
    st.line_chart(df["energy"].iloc[-500:])

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 3
# ==================================================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Next 24-Hour Forecast")

    forecast_df = create_next_24_forecast(input_df)

    f1, f2, f3 = st.columns(3)
    f1.metric("Next Hour Forecast", f"{forecast_df['predicted_energy'].iloc[0]:.2f} MW")
    f2.metric("24h Average Forecast", f"{forecast_df['predicted_energy'].mean():.2f} MW")
    f3.metric("24h Max Forecast", f"{forecast_df['predicted_energy'].max():.2f} MW")

    st.line_chart(forecast_df.set_index("forecast_hour_ahead")["predicted_energy"])
    st.dataframe(forecast_df, width="stretch")

    if forecast_df["predicted_energy"].max() >= 2500:
        st.warning("At least one high-demand period is expected in the next 24 hours.")
    else:
        st.success("No extreme high-demand period is expected in the next 24 hours.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 4
# ==================================================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Scenario Comparison")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("### Scenario A")
        hour_a = st.slider("Hour A", 0, 23, 10)
        lag_1_a = st.number_input("Last Hour Energy A", value=lag_1, key="lag_a")
        rolling_24_a = st.number_input("24h Average A", value=rolling_24, key="roll_a")

    with col_b:
        st.write("### Scenario B")
        hour_b = st.slider("Hour B", 0, 23, 18)
        lag_1_b = st.number_input("Last Hour Energy B", value=lag_1 + 200, key="lag_b")
        rolling_24_b = st.number_input("24h Average B", value=rolling_24 + 150, key="roll_b")

    scenario_a = input_df.copy()
    scenario_b = input_df.copy()

    scenario_a["hour"] = hour_a
    scenario_a["lag_1"] = lag_1_a
    scenario_a["rolling_24"] = rolling_24_a

    scenario_b["hour"] = hour_b
    scenario_b["lag_1"] = lag_1_b
    scenario_b["rolling_24"] = rolling_24_b

    pred_a = model.predict(scenario_a)[0]
    pred_b = model.predict(scenario_b)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Scenario A Prediction", f"{pred_a:.2f} MW")
    c2.metric("Scenario B Prediction", f"{pred_b:.2f} MW")
    c3.metric("Difference", f"{abs(pred_b - pred_a):.2f} MW")

    scenario_plot_df = pd.DataFrame({
        "Scenario": ["Scenario A", "Scenario B"],
        "Predicted Energy": [pred_a, pred_b]
    })
    st.bar_chart(scenario_plot_df.set_index("Scenario"))

    if pred_b > pred_a:
        st.warning("Scenario B has higher predicted energy consumption.")
    elif pred_a > pred_b:
        st.info("Scenario A has higher predicted energy consumption.")
    else:
        st.success("Both scenarios have similar predicted energy consumption.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 5
# ==================================================
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Average Energy", f"{df['energy'].mean():.2f} MW")

    st.write("### First Rows")
    st.dataframe(df.head(), width="stretch")

    st.write("### Statistical Summary")
    st.dataframe(df.describe(), width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("EDA Visualizations")

    img_cols = st.columns(2)

    visuals = [
        ("energy_distribution_full.png", "Energy Distribution"),
        ("energy_boxplot.png", "Energy Boxplot"),
        ("avg_energy_by_hour_full.png", "Average Energy by Hour"),
        ("avg_energy_by_day_of_week.png", "Average Energy by Day of Week"),
        ("avg_energy_by_month_full.png", "Average Energy by Month"),
        ("correlation_matrix_energy.png", "Correlation Matrix"),
        ("energy_time_series_sample.png", "Energy Time-Series Sample"),
        ("missing_values_energy.png", "Missing Values")
    ]

    for idx, (file, caption) in enumerate(visuals):
        if Path(file).exists():
            with img_cols[idx % 2]:
                st.image(file, caption=caption, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 6
# ==================================================
with tab6:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Comparison")

    if results_df is not None:
        st.dataframe(results_df, width="stretch")

    if Path("energy_model_comparison.png").exists():
        st.image("energy_model_comparison.png", caption="Model Comparison by Test R²", width="stretch")

    st.subheader("Final Selected Model")
    st.write("""
    **Final Model:** Tuned Decision Tree Regressor

    **Why this model was selected:**
    - It achieved the highest test R² score.
    - It reduced prediction error compared with earlier models.
    - It captures non-linear time-series patterns better than linear regression.
    - Tuning reduced overfitting while maintaining strong performance.
    """)

    st.write("""
    **Final Performance**
    - Train R²: 0.9952
    - Test R²: 0.9910
    - Test MAE: 25.37 MW
    - Test RMSE: 35.69 MW
    """)

    st.subheader("Feature Importance")
    if importance_df is not None:
        st.dataframe(importance_df, width="stretch")
        st.bar_chart(importance_df.set_index("Feature"))

    if Path("final_feature_importance.png").exists():
        st.image("final_feature_importance.png", caption="Final Feature Importance", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 7
# ==================================================
with tab7:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Error Analysis")

    if pred_df is not None:
        st.write("### Prediction Samples")
        st.dataframe(pred_df.head(20), width="stretch")

        if "absolute_error" in pred_df.columns:
            e1, e2, e3 = st.columns(3)
            e1.metric("Mean Absolute Error", f"{pred_df['absolute_error'].mean():.2f} MW")
            e2.metric("Max Absolute Error", f"{pred_df['absolute_error'].max():.2f} MW")
            e3.metric("Median Absolute Error", f"{pred_df['absolute_error'].median():.2f} MW")

            st.write("### Top 10 Highest Absolute Errors")
            st.dataframe(pred_df.sort_values(by="absolute_error", ascending=False).head(10), width="stretch")

    if Path("final_actual_vs_predicted_line.png").exists():
        st.image("final_actual_vs_predicted_line.png", caption="Actual vs Predicted Energy Over Time", width="stretch")

    if Path("final_error_distribution.png").exists():
        st.image("final_error_distribution.png", caption="Final Error Distribution", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Peak vs Non-Peak Analysis")

    peak_hours = [17, 18, 19, 20]
    df_peak = df.copy()
    df_peak["is_peak_hour"] = df_peak["hour"].isin(peak_hours).astype(int)

    peak_summary = df_peak.groupby("is_peak_hour")["energy"].agg(["mean", "min", "max", "std"]).reset_index()
    peak_summary["Period"] = peak_summary["is_peak_hour"].map({0: "Non-Peak", 1: "Peak"})
    peak_summary = peak_summary[["Period", "mean", "min", "max", "std"]]

    st.dataframe(peak_summary, width="stretch")
    st.bar_chart(peak_summary.set_index("Period")[["mean"]])

    st.write("""
    Peak-hour analysis helps evaluate whether energy consumption increases during evening demand periods.
    This supports the business interpretation of the forecasting model.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# TAB 8
# ==================================================
with tab8:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Download Project Outputs")

    st.download_button(
        "Download Advanced Cleaned Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "advanced_cleaned_energy.csv",
        "text/csv"
    )

    if pred_df is not None:
        st.download_button(
            "Download Final Predictions",
            pred_df.to_csv(index=False).encode("utf-8"),
            "final_energy_predictions.csv",
            "text/csv"
        )

    if results_df is not None:
        st.download_button(
            "Download Model Results",
            results_df.to_csv(index=False).encode("utf-8"),
            "advanced_energy_model_results.csv",
            "text/csv"
        )

    if importance_df is not None:
        st.download_button(
            "Download Feature Importance",
            importance_df.to_csv(index=False).encode("utf-8"),
            "final_feature_importance.csv",
            "text/csv"
        )

    forecast_df = create_next_24_forecast(input_df)
    st.download_button(
        "Download 24-Hour Forecast",
        forecast_df.to_csv(index=False).encode("utf-8"),
        "next_24_hour_forecast.csv",
        "text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)
