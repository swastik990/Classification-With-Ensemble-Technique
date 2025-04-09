import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
@st.cache_resource
def load_model():
    model_path = 'model.joblib'
    model = joblib.load(model_path)
    return model

model = load_model()

# Title of the app
st.title("Loan Approval Prediction")

# Description
st.write("""
This app predicts whether a loan will be approved or not based on user input.
Please provide the required details below.
""")

# Input fields for features
st.header("Enter Loan Details")

# Numerical features
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20)
income_annum = st.number_input("Annual Income (in USD)", min_value=0)
loan_amount = st.number_input("Loan Amount (in USD)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value (in USD)", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value (in USD)", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value (in USD)", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value (in USD)", min_value=0)

# Categorical features
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# Button to trigger prediction
if st.button("Predict"):
    # Convert categorical features to numeric values
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    # Prepare input data as a NumPy array
    input_data = pd.DataFrame([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]], columns=[
            ' no_of_dependents',
            ' education',
            ' self_employed',
            ' income_annum',
            ' loan_amount',
            ' loan_term',
            ' cibil_score',
            ' residential_assets_value',
            ' commercial_assets_value',
            ' luxury_assets_value',
            ' bank_asset_value'
        ])

    # Make predictions
    prediction = model.predict(input_data)

    # Display the result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("Congratulations! Your loan is likely to be approved.")
    else:
        st.error("Sorry, your loan is unlikely to be approved.")