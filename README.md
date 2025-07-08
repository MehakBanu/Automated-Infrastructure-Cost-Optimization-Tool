# Automated-Infrastructure-Cost-Optimization-Tool
This project is a Streamlit-based web application that analyzes cloud infrastructure usage data (CPU usage, memory usage, and cost per day), detects anomalies using machine learning, predicts daily costs based on resource usage, and provides actionable recommendations to optimize infrastructure cost. The tool is designed to help cloud users make intelligent decisions about resizing, reservation, and anomaly handling in a simple and interactive way.

Features:

Upload cloud usage data in CSV format.

Detect anomalies in cost using Isolation Forest.

Get recommendations such as:

"Consider downsizing instance"

"Switch to reserved instance"

"No action needed"

Predict daily cost using a trained Random Forest model based on CPU and memory inputs.

Download the full analysis and suggestions as a CSV report.

Input Requirements:
The input file should be a CSV file with the following columns:

date (in YYYY-MM-DD format)

cpu_usage (0–100 scale)

memory_usage (0–100 scale)

cost_per_day (numeric)

How to Run the Project:

Clone the GitHub repository to your local machine:
git clone https://github.com/MehakBanu/Automated-Infrastructure-Cost-Optimization-Tool.git

Navigate to the project folder:
cd Automated-Infrastructure-Cost-Optimization-Tool

Install the required dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Machine Learning Details:

Anomaly detection is handled using IsolationForest from scikit-learn.

Cost prediction is handled using a RandomForestRegressor.

Model performance is shown with R² Score and Mean Absolute Error (MAE).

Project Files:

app.py – Main Streamlit application

requirements.txt – Python packages required

sample_data.csv – Example CSV input file
