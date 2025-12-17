import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Advertising Sales Prediction")

st.title("📊 Advertising Sales Prediction")
st.write("Polynomial Regression Model")

# Load pickle
with open("advertising_poly_model.pkl", "rb") as file:
    loaded_obj = pickle.load(file)

# Detect model type
if isinstance(loaded_obj, tuple):
    model, poly = loaded_obj
else:
    model = loaded_obj
    poly = None

# Inputs
tv = st.number_input("TV Advertising Budget", min_value=0.0, step=1.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0, step=1.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, step=1.0)

if st.button("Predict Sales"):
    X = np.array([[tv, radio, newspaper]])

    # Apply polynomial transformation if needed
    if poly is not None:
        X = poly.transform(X)

    prediction = model.predict(X)
    st.success(f"📈 Predicted Sales: {prediction[0]:.2f}units")
