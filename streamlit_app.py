import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model safely
@st.cache_resource
def load_model():
    model = joblib.load("decision_tree_model.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, features

model, feature_cols = load_model()

st.title("💰 Customer Sales Prediction App")
st.write("Fill the details below to predict Total Sales for a customer.")

# ----- USER INPUT FIELDS -----

total_invoices = st.number_input("Total Invoices (Frequency)", min_value=0, step=1)
recency = st.number_input("Recency (Days Since Last Purchase)", min_value=0, step=1)
avg_order_value = st.number_input("Average Order Value", min_value=0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
country = st.text_input("Country (e.g., United Kingdom)")

# Button
if st.button("Predict Sales"):
    # Create input row
    input_dict = {
        "TotalInvoices": total_invoices,
        "Recency": recency,
        "AvgOrderValue": avg_order_value,
        "Gender": gender,
        "Country": country
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Align with training features
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Model prediction (log)
    y_pred_log = model.predict(input_df)[0]

    # Convert back to normal sales value
    y_pred = np.expm1(y_pred_log)

    st.success(f"Predicted Total Sales: **${y_pred:.2f}**")