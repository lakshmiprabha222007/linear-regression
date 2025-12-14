import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("trained_linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Linear Regression Prediction App")

st.write("Enter input values to get prediction")

# Example: single feature input
x = st.number_input("Enter feature value", value=0.0)

if st.button("Predict"):
    input_data = np.array([[x]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")

