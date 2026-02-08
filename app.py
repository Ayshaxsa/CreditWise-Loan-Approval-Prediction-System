import streamlit as st
import pandas as pd
import joblib  # your model was saved with joblib

# Load the trained Naive Bayes model
model = joblib.load("naive_bayes_model.pkl")

st.set_page_config(page_title="CreditWise Loan System")

st.title("💳 CreditWise Loan Approval System")
st.write("Enter applicant details to predict loan approval.")

# User inputs matching your dataset
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
income = st.number_input("Applicant Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)
credit_history = st.selectbox("Credit History", [1, 0])

# Create dataframe from user input
input_df = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Education': [education],
    'ApplicantIncome': [income],
    'LoanAmount': [loan_amount],
    'Credit_History': [credit_history]
})

# Preprocess input (must match training preprocessing exactly)
def preprocess(df):
    df = df.copy()
    # Encode categorical columns
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    # Ensure numeric columns have correct types
    df['ApplicantIncome'] = df['ApplicantIncome'].astype(float)
    df['LoanAmount'] = df['LoanAmount'].astype(float)
    df['Credit_History'] = df['Credit_History'].astype(int)
    # Reorder columns exactly as in training
    df = df[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
    return df

processed_input = preprocess(input_df)

# Prediction
prediction = model.predict(processed_input)[0]
prediction_proba = model.predict_proba(processed_input)

st.subheader("Prediction")
st.write("Approved ✅" if prediction == 1 else "Rejected ❌")

st.subheader("Prediction Probability")
st.write(prediction_proba)
