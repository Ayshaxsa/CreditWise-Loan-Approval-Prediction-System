import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("naive_bayes_model.pkl")
ohe = joblib.load("ohe_encoder.pkl")

st.set_page_config(page_title="CreditWise Loan Approval System")
st.title("💳 CreditWise Loan Approval System")

# ------------------------------
# User Inputs
# ------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
loan_purpose = st.selectbox("Loan Purpose", ["Car", "Home", "Education", "Personal"])
employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
employer_category = st.selectbox("Employer Category", ["Private", "MNC", "Government", "Unemployed"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
age = st.number_input("Age", min_value=18)
dependents = st.number_input("Dependents", min_value=0)
existing_loans = st.number_input("Existing Loans", min_value=0)
savings = st.number_input("Savings", min_value=0)
collateral_value = st.number_input("Collateral Value", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term", min_value=1)

# ------------------------------
# Create DataFrame
# ------------------------------
input_df = pd.DataFrame({
    'Marital_Status': [married],
    'Loan_Purpose': [loan_purpose],
    'Employment_Status': [employment_status],
    'Property_Area': [property_area],
    'Gender': [gender],
    'Employer_Category': [employer_category],
    'Education_Level': [education],
    'Applicant_Income': [applicant_income],
    'Coapplicant_Income': [coapplicant_income],
    'Age': [age],
    'Dependents': [dependents],
    'Existing_Loans': [existing_loans],
    'Savings': [savings],
    'Collateral_Value': [collateral_value],
    'Loan_Amount': [loan_amount],
    'Loan_Term': [loan_term]
})

# ------------------------------
# Encode categorical inputs with manual mapping
# ------------------------------
# Map Education_Level manually
education_map = {"Graduate": 1, "Not Graduate": 0}
input_df['Education_Level'] = input_df['Education_Level'].map(education_map)

# Encode other categorical columns using saved OneHotEncoder
categorical_cols = ['Marital_Status', 'Loan_Purpose', 'Employment_Status', 'Property_Area', 'Gender', 'Employer_Category']
encoded_cols = ohe.transform(input_df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(categorical_cols), index=input_df.index)

# Drop original categorical columns and add encoded columns
input_df = pd.concat([input_df.drop(categorical_cols, axis=1), encoded_df], axis=1)

# ------------------------------
# Make prediction
# ------------------------------
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# ------------------------------
# Show results
# ------------------------------
st.subheader("Prediction")
st.write("✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected")

st.subheader("Prediction Probability")
st.write(prediction_proba)
