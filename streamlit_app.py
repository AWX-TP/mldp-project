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

    # Create input DataFrame
    input_df = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Apply same preprocessing
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Input validation
    if total_charges < monthly_charges:
        st.warning("Total Charges should usually be greater than Monthly Charges.")
        st.stop()

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
