import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="🔮",
    layout="wide"
)

# Title and description
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("""
This application predicts whether a customer is likely to **Churn** using a trained **Gaussian Naive Bayes** machine learning model.
Fill in the customer details in the sidebar to run a real-time risk assessment.
""")

# Load saved artifacts
@st.cache_resource
def load_artifacts():
    # Attempt to load model, scaler, features, and metadata
    model = joblib.load('model.pkl') if os.path.exists('model.pkl') else None
    scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
    feature_names = joblib.load('feature_names.pkl') if os.path.exists('feature_names.pkl') else None
    metadata = joblib.load('metadata.pkl') if os.path.exists('metadata.pkl') else None
    return model, scaler, feature_names, metadata

model, scaler, feature_names, metadata = load_artifacts()

# Display metadata metric if available
if metadata and isinstance(metadata, dict):
    st.sidebar.info(f"**Model:** {metadata.get('Model', 'GaussianNB')} | **Target:** {metadata.get('Target', 'Churn')}")

st.sidebar.header("📋 Customer Input Features")

# --- Input Fields ---
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges), step=10.0)

# --- Feature Engineering (Matches exact training pipeline features) ---
# 1. Tenure Grouping
if tenure <= 12:
    tenure_group = '0-1 year'
elif tenure <= 24:
    tenure_group = '1-2 years'
elif tenure <= 48:
    tenure_group = '2-4 years'
else:
    tenure_group = '4-6 years'

# 2. Monthly Charges Category
if monthly_charges <= 35:
    monthly_cat = 'Low'
elif monthly_charges <= 70:
    monthly_cat = 'Medium'
else:
    monthly_cat = 'High'

# 3. Total Charges Category
if total_charges <= 500:
    total_cat = 'Low'
elif total_charges <= 3000:
    total_cat = 'Medium'
else:
    total_cat = 'High'

# 4. Additional Derived Metrics
avg_monthly_spend = total_charges / tenure if tenure > 0 else monthly_charges
services = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
service_count = sum(1 for s in services if s == "Yes")
engagement_score = service_count + (1 if contract != "Month-to-month" else 0)

long_term_customer = 1 if tenure > 24 else 0
high_monthly_charges = 1 if monthly_charges > 70 else 0
paperless_billing_flag = 1 if paperless_billing == "Yes" else 0
senior_citizen_cat = "Senior" if senior_citizen == 1 else "Non-Senior"

# --- Build Input DataFrame ---
raw_input = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'TenureGroup': tenure_group,
    'MonthlyChargesCategory': monthly_cat,
    'TotalChargesCategory': total_cat,
    'AverageMonthlySpend': avg_monthly_spend,
    'EngagementScore': engagement_score,
    'ServiceCount': service_count,
    'LongTermCustomer': long_term_customer,
    'HighMonthlyCharges': high_monthly_charges,
    'PaperlessBillingFlag': paperless_billing_flag,
    'SeniorCitizenCategory': senior_citizen_cat
}

input_df = pd.DataFrame([raw_input])

# --- Preprocessing & Encoding ---
# Expected 27 features from training artifacts
expected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
    'TotalCharges', 'TenureGroup', 'MonthlyChargesCategory',
    'TotalChargesCategory', 'AverageMonthlySpend', 'EngagementScore',
    'ServiceCount', 'LongTermCustomer', 'HighMonthlyCharges',
    'PaperlessBillingFlag', 'SeniorCitizenCategory'
]

# Ensure input DataFrame matches the exact column list and order
encoded_df = input_df.copy()
for col in encoded_df.columns:
    if encoded_df[col].dtype == 'object' or isinstance(encoded_df[col].dtype, pd.CategoricalDtype):
        encoded_df[col] = encoded_df[col].astype('category').cat.codes

# Reindex columns to match feature order expected by scaler/model
if feature_names is not None:
    encoded_df = encoded_df.reindex(columns=list(feature_names), fill_value=0)

# Display User Summary Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Contract", contract)
col2.metric("Tenure", f"{tenure} mos ({tenure_group})")
col3.metric("Monthly Charge", f"${monthly_charges:.2f}")
col4.metric("Active Services", f"{service_count} Services")

st.divider()

# --- Prediction Action ---
if st.button("🚀 Predict Churn Probability", use_container_width=True):
    if model is not None and scaler is not None:
        try:
            # Scale numerical inputs
            scaled_features = scaler.transform(encoded_df)
            
            # Predict Class and Probabilities
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            
            churn_prob = probabilities[1] * 100
            retained_prob = probabilities[0] * 100

            st.subheader("📊 Prediction Results")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediction == 1:
                    st.error("🚨 **High Risk of Churn**")
                else:
                    st.success("✅ **Likely to Retain**")
                
                st.metric(label="Churn Probability", value=f"{churn_prob:.1f}%")

            with res_col2:
                st.write("**Probability Breakdown**")
                st.progress(float(probabilities[1]))
                st.write(f"- **Retention Likelihood:** {retained_prob:.1f}%")
                st.write(f"- **Churn Likelihood:** {churn_prob:.1f}%")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("⚠️ Model or Scaler files (`model.pkl`, `scaler.pkl`) were not found in the root directory. Please verify file paths.")

# Display raw input data for transparency
with st.expander("🔎 View Raw Input DataFrame"):
    st.dataframe(input_df)
