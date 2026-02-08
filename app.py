import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="CreditWise Loan System")

st.title("💳 CreditWise Loan Approval System")
st.write("Enter applicant details to predict loan approval.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=70, value=30)
income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)
loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)

if st.button("Predict Loan Approval"):
    # Create input array
    input_data = np.array([[age, income, credit_score, loan_amount, loan_term]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
