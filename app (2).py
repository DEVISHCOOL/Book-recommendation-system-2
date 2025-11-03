# ------------------------------------------
# XGBoost Deployment using Streamlit
# ------------------------------------------

import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.title("⚡ XGBoost Model Deployment")
st.write("Enter feature values below to get predictions from your trained XGBoost model.")

# -------------------------------
# Load trained XGBoost model
# -------------------------------
# If saved with pickle:
# with open("xgb_model.pkl", "rb") as file:
#     model = pickle.load(file)

# If saved as native XGBoost model:
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

st.success("✅ Model loaded successfully!")

# -------------------------------
# User input for prediction
# -------------------------------
# Replace these example features with your actual input features
st.subheader("Enter input features:")
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# Combine user inputs into a DataFrame
input_data = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                          columns=['feature1', 'feature2', 'feature3', 'feature4'])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.subheader("🧾 Prediction Result:")
        st.write(f"Predicted Class: {prediction[0]}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

