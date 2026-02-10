import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load trained model and feature columns
model = joblib.load("churn_best_rf_model.pkl")
model_features = joblib.load("feature_columns.pkl")

## Streamlit app
st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using a trained Random Forest model.")

st.header("Enter Customer Information")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=5)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.85)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=354.25)

if st.button("Predict Churn"):

    errors = []

    # Validation rules
    if tenure <= 0:
        errors.append("Tenure must be greater than 0 months.")

    if monthly_charges <= 0:
        errors.append("Monthly Charges must be greater than $0.")

    if total_charges <= 0:
        errors.append("Total Charges must be greater than $0.")

    if tenure > 0 and total_charges < (monthly_charges * tenure * 0.5):
        errors.append(
            "Total Charges is unrealistically low compared to Tenure and Monthly Charges."
        )

    # Display errors if any
    if errors:
        st.error("Please correct the following input errors:")
        for err in errors:
            st.write(f"- {err}")
    else:
        # Create input DataFrame
        input_df = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        # Apply same preprocessing
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Predict
        try:
            with st.spinner("Predicting churn risk..."):
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.stop()

        # Display result
        if prediction == 1:
            st.error(f"High Risk of Churn (Probability: {probability:.2f})")
        else:
            st.success(f"Likely to Stay (Churn Probability: {probability:.2f})")
