
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("loan_approval_scaler.pkl")

st.title("Loan Approval Prediction")
st.write("Machine Learning Based Loan Assessment System")

st.subheader("Enter Applicant Information")

no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0.0,
    step=100000.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    step=100000.0
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1,
    step=1
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700,
    step=1
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0.0,
    step=100000.0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0.0,
    step=100000.0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0.0,
    step=100000.0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0.0,
    step=100000.0
)

if st.button("Predict Loan Approval"):

    input_data = pd.DataFrame([[
        no_of_dependents,
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=[
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.success("Loan Status: APPROVED")
    else:
        st.error("Loan Status: REJECTED")

    st.write(f"Approval Probability: **{probability:.2f}%**")
