import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("advertising_poly_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Advertising Sales Prediction")

st.title("📊 Advertising Sales Prediction")
st.write("Polynomial Regression Model")

# Input fields
tv = st.number_input("TV Advertising Budget", min_value=0.0, step=1.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0, step=1.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, step=1.0)

# Predict button
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)

    st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")
