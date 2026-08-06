import streamlit as st
import pandas as pd
import joblib
import os

# Load Model

linear = joblib.load("linear_model (1).pkl")
poly_model = joblib.load("polynomial_model (1).pkl")
poly = joblib.load("polynomial_features (1).pkl")

# Page Configuration

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼")

st.title("💼 Employee Salary Prediction")

# Inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education Level",
    [
        "Bachelor's",
        "Master's",
        "PhD"
    ]
)

job = st.selectbox(
    "Job Title",
    [
        "Software Engineer",
        "Data Analyst",
        "Manager",
        "Sales Associate",
        "Senior Manager",
        "Director",
        "Marketing Analyst",
        "Product Manager",
        "HR Manager"
    ]
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=5.0
)

# Encode Inputs
gender_encoded = 1 if gender == "Male" else 0

education_map = {
    "Bachelor's": 0,
    "Master's": 1,
    "PhD": 2
}
education_encoded = education_map[education]

job_map = {
    "Software Engineer": 0,
    "Data Analyst": 1,
    "Manager": 2,
    "Sales Associate": 3,
    "Senior Manager": 4,
    "Director": 5,
    "Marketing Analyst": 6,
    "Product Manager": 7,
    "HR Manager": 8
}
job_title_encoded = job_map[job]

# Select Model
model_choice = st.selectbox(
    "Choose Prediction Model",
    ["Linear Regression", "Polynomial Regression"]
)

# Predict
if st.button("Predict Salary"):

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Education Level": [education_encoded],
        "Job Title": [job_title_encoded],
        "Years of Experience": [experience]
    })

    if model_choice == "Linear Regression":
        prediction = linear.predict(input_df)

    else:
        input_poly = poly.transform(input_df)
        prediction = poly_model.predict(input_poly)

    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")

