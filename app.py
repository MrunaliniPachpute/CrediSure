import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="CrediSure-Loan System", layout="wide")

model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏦 CrediSure - Loan Approval System")
st.markdown("### Intelligent ML-based Loan Approval Predictor")

# layout - 2cols
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Financial Details")
    Applicant_Income = st.number_input("Applicant Income", min_value=0)
    Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
    Savings = st.number_input("Savings", min_value=0)
    Collateral_Value = st.number_input("Collateral Value", min_value=0)
    Loan_Amount = st.number_input("Loan Amount", min_value=0)
    Loan_Term = st.number_input("Loan Term (months)", min_value=0)

with col2:
    st.subheader("👤 Personal & Risk Profile")
    Age = st.number_input("Age", min_value=18, max_value=100)
    Dependents = st.number_input("Dependents", min_value=0)
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900)
    Existing_Loans = st.number_input("Existing Loans", min_value=0)
    DTI_Ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)

# categorical
st.subheader("📌 Additional Details")

col3, col4 = st.columns(2)

with col3:
    Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Business", "Contract", "Unemployed"])
    Marital_Status = st.selectbox("Marital Status", ["Married", "Single"])

with col4:
    Loan_Purpose = st.selectbox("Loan Purpose", ["Business", "Education", "Home", "Personal", "Car"])
    Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Employer_Category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])

# Processing
data = {
    "Applicant_Income": Applicant_Income,
    "Coapplicant_Income": Coapplicant_Income,
    "Age": Age,
    "Dependents": Dependents,
    "Credit_Score": Credit_Score,
    "Existing_Loans": Existing_Loans,
    "DTI_Ratio": DTI_Ratio,
    "Savings": Savings,
    "Collateral_Value": Collateral_Value,
    "Loan_Amount": Loan_Amount,
    "Loan_Term": Loan_Term,
    "Education_Level": 1 if Education_Level == "Not Graduate" else 0,
}

all_possible_cols = [
    "Employment_Status_Self-employed","Employment_Status_Unemployed","Employment_Status_Business","Employment_Status_Contract",
    "Marital_Status_Single",
    "Loan_Purpose_Education","Loan_Purpose_Home","Loan_Purpose_Personal","Loan_Purpose_Car",
    "Property_Area_Semiurban","Property_Area_Urban",
    "Gender_Male",
    "Employer_Category_Government","Employer_Category_MNC","Employer_Category_Private","Employer_Category_Unemployed"
]

for col in all_possible_cols:
    data[col] = 0

if Employment_Status != "Salaried":
    data[f"Employment_Status_{Employment_Status}"] = 1
if Marital_Status == "Single":
    data["Marital_Status_Single"] = 1
if Loan_Purpose != "Business":
    data[f"Loan_Purpose_{Loan_Purpose}"] = 1
if Property_Area != "Rural":
    data[f"Property_Area_{Property_Area}"] = 1
if Gender == "Male":
    data["Gender_Male"] = 1
if Employer_Category != "Private":
    data[f"Employer_Category_{Employer_Category}"] = 1

# FE
data["DTI_Ratio_sq"] = DTI_Ratio ** 2
data["Credit_Score_sq"] = Credit_Score ** 2
data["Applicant_Income_Log"] = np.log1p(Applicant_Income)

data.pop("DTI_Ratio")
data.pop("Credit_Score")

input_df = pd.DataFrame([data])
input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

input_scaled = scaler.transform(input_df)

# Predictions
if st.button("🚀 Predict Loan Approval"):
    result = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    if result[0] == 1:
        st.success(f"✅ Loan Approved")
        st.balloons()
    else:
        st.error(f"❌ Loan Rejected")