
"""
Telco Customer Churn Prediction - Streamlit App
Generated template.

IMPORTANT:
This template loads your saved model artifacts:
- churn_model.pkl
- scaler.pkl
- feature_names.pkl

Replace the FEATURE_ENGINEERING() function with the exact logic from
your training notebook if you want predictions to exactly match training.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Customer Churn", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

st.title("📊 Telco Customer Churn Prediction")

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female","Male"])
senior = st.sidebar.selectbox("Senior Citizen", ["No","Yes"])
partner = st.sidebar.selectbox("Partner", ["No","Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No","Yes"])
tenure = st.sidebar.slider("Tenure",0,72,12)
monthly = st.sidebar.number_input("Monthly Charges",0.0,200.0,70.0)
total = st.sidebar.number_input("Total Charges",0.0,10000.0,1000.0)

def FEATURE_ENGINEERING():
    """
    Replace this function with the exact preprocessing from your notebook.
    """
    row = {c:0 for c in feature_names}
    for k,v in {
        "tenure":tenure,
        "MonthlyCharges":monthly,
        "TotalCharges":total
    }.items():
        if k in row:
            row[k]=v
    return pd.DataFrame([row], columns=feature_names)

if st.button("Predict"):
    X = FEATURE_ENGINEERING()
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    proba = model.predict_proba(Xs)[0]
    if pred==1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")
    st.metric("Churn Probability", f"{proba[1]*100:.2f}%")
    st.metric("Stay Probability", f"{proba[0]*100:.2f}%")
