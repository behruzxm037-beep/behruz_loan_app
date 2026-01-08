import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1️⃣ Model va preprocessorsni yuklash
# -----------------------------
knn_model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le_dict = joblib.load("models/le_dict.pkl")

# -----------------------------
# 2️⃣ Streamlit UI - Inputlar
# -----------------------------
st.title("Bank Loan Decision Predictor")
st.write("Loan olish qarorini bashorat qilish uchun barcha ma'lumotlarni kiriting:")

# Numeric inputlar
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
income_annum = st.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
loan_term = st.number_input("Loan Term (months)", min_value=1, value=36)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=100000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=50000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=20000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=30000)

# Categorical inputlar
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# -----------------------------
# 3️⃣ Prediction tugmasi
# -----------------------------
if st.button("Predict Loan Decision"):

    # DataFrame yaratish
    input_df = pd.DataFrame({
        "education": [education],
        "self_employed": [self_employed],
        "no_of_dependents": [no_of_dependents],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]
    })

    # Categorical ustunlarni LabelEncoder bilan transform qilish
    for col in ["education", "self_employed"]:
        input_df[col] = le_dict[col].transform(input_df[col])

    # Numeric ustunlarni scaler bilan transform qilish
    X_scaled = scaler.transform(input_df)

    # Bashorat
    prediction = knn_model.predict(X_scaled)[0]

    # Natijani chiqarish
    result_text = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
    st.subheader("Prediction Result:")
    st.write(result_text)
