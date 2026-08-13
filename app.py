import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load trained model and preprocessing objects
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("onehot_encoder.pkl")


# Page configuration
st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide"
)


# Title
st.title("💳 CreditWise")
st.subheader("Loan Approval Prediction System")

st.write(
    "Enter the applicant's details below to predict whether "
    "the loan is likely to be approved."
)

# --------------------------------------------------
# Loan Application Form
# --------------------------------------------------

st.header("📋 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        value=0,
        step=1
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=10000.0
    )

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=20000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        value=36,
        step=1
    )

with col2:
    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        value=0.30
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0.0,
        max_value=1000.0,
        value=700.0
    )

    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        [
            "Contract",
            "Salaried",
            "Self-employed",
            "Unemployed"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Business",
            "Car",
            "Education",
            "Home",
            "Personal"
        ]
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Rural",
            "Semiurban",
            "Urban"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        [
            "Business",
            "Government",
            "MNC",
            "Private",
            "Unemployed"
        ]
    )

    # --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = None

if st.button("🔍 Predict Loan Approval", use_container_width=True):

    # Create dataframe from user input
    input_data = pd.DataFrame({
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Age": [age],
        "Dependents": [dependents],
        "Existing_Loans": [existing_loans],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Education_Level": [education_level],
        "Employment_Status": [employment_status],
        "Marital_Status": [marital_status],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Gender": [gender],
        "Employer_Category": [employer_category],
        "DTI_Ratio": [dti_ratio],
        "Credit_Score": [credit_score]
    })

    # Education encoding
    education_mapping = {
        "Graduate": 0,
        "Not Graduate": 1
    }

    input_data["Education_Level"] = (
        input_data["Education_Level"].map(education_mapping)
    )

    # One-hot encoding
    categorical_features = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    encoded_data = ohe.transform(
        input_data[categorical_features]
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ohe.get_feature_names_out(categorical_features)
    )

    input_data = input_data.drop(
        columns=categorical_features
    )

    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Feature engineering
    input_data["DTI_Ratio_sq"] = input_data["DTI_Ratio"] ** 2
    input_data["Credit_Score_sq"] = input_data["Credit_Score"] ** 2
    input_data["Applicant_Income_log"] = np.log1p(
        input_data["Applicant_Income"]
    )

    # Remove original features
    input_data = input_data.drop(
        columns=["DTI_Ratio", "Credit_Score"]
    )

    # Load exact feature order
    feature_columns = joblib.load("feature_columns.pkl")

    input_data = input_data[feature_columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]


# Display result only if prediction exists
if prediction is not None:

    if prediction == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected")

    st.caption(
        "Prediction generated using Gaussian Naive Bayes "
        "based on the trained CreditWise model."
    )

st.markdown("---")

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Precision", "81.13%")

with col2:
    st.metric("Recall", "70.49%")

with col3:
    st.metric("F1 Score", "75.44%")

with col4:
    st.metric("Accuracy", "86.00%")

st.info(
    "⚠️ This prediction is for educational and demonstration purposes "
    "only and should not be used as an actual lending decision."
)