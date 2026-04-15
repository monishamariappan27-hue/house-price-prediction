import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Train model
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

# UI
st.title("🏠 House Price Prediction")

area = st.number_input("Enter Area (sq ft)")
bedrooms = st.number_input("Number of Bedrooms")
bathrooms = st.number_input("Number of Bathrooms")

if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")