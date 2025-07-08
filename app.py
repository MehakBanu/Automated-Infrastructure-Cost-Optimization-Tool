import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Cloud Cost Optimization", layout="centered")
st.title("☁️ Automated Infrastructure Cost Optimization Tool")

st.markdown("""
This tool analyzes cloud usage patterns (CPU, memory, cost), detects anomalies, 
predicts future costs using ML, and suggests cost-saving actions.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload cloud usage CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file, parse_dates=["date"])

    # --- Anomaly Detection ---
    def detect_anomalies(df):
        model = IsolationForest(contamination=0.1, random_state=42)
        df['anomaly'] = model.fit_predict(df[['cpu_usage', 'memory_usage', 'cost_per_day']])
        df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
        return df

    # --- Optimization Suggestions ---
    def suggest_optimizations(df):
        suggestions = []
        for _, row in df.iterrows():
            if row['cost_per_day'] > 130 and row['cpu_usage'] < 50:
                suggestions.append("Consider downsizing instance")
            elif row['cost_per_day'] > 150 and row['cpu_usage'] > 85:
                suggestions.append("Switch to reserved instance")
            elif row['anomaly'] == 'Anomaly':
                suggestions.append("Investigate unusual spike")
            else:
                suggestions.append("No action needed")
        df['suggestion'] = suggestions
        return df

    # --- Train Cost Prediction Model ---
    def train_cost_predictor(df):
        X = df[['cpu_usage', 'memory_usage']]
        y = df['cost_per_day']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mae, r2

    # --- Process and Display ---
    with st.spinner("🔍 Analyzing..."):
        df = detect_anomalies(df)
        df = suggest_optimizations(df)
        model, mae, r2 = train_cost_predictor(df)

    st.subheader("📊 Optimization Report")
    st.dataframe(df, use_container_width=True)

    # --- Cost Prediction Tool ---
    st.subheader("🔮 Predict Cost Based on Usage")
    cpu_input = st.slider("CPU Usage (%)", 0, 100, 60)
    memory_input = st.slider("Memory Usage (%)", 0, 100, 70)
    input_df = pd.DataFrame({"cpu_usage": [cpu_input], "memory_usage": [memory_input]})
    predicted_cost = model.predict(input_df)[0]
    st.success(f"💡 Predicted Daily Cost: ₹{predicted_cost:.2f}")

    # --- Model Evaluation ---
    st.markdown("#### 🧠 ML Model Performance")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # --- Download Button ---
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Report", data=csv, file_name="cost_optimization_report.csv", mime="text/csv")

else:
    st.warning("📁 Please upload a CSV file to begin analysis.")
