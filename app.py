import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Customer Churn Dashboard", layout="wide")

model = joblib.load("bank_churn_model.pkl")
feature_importance = pd.read_csv("feature_importance.csv")

st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")

st.sidebar.header("Customer Input")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 300000.0, 70000.0)

balance_salary_ratio = balance / (salary + 1)
product_density = num_products / (tenure + 1)
engagement_product_interaction = is_active * num_products
age_tenure_interaction = age * tenure

input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [salary],
    "BalanceSalaryRatio": [balance_salary_ratio],
    "ProductDensity": [product_density],
    "EngagementProductInteraction": [engagement_product_interaction],
    "AgeTenureInteraction": [age_tenure_interaction]
})

st.subheader("Customer Input Data")
st.write(input_df)

prob = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.write(f"Churn Probability: {prob:.2%}")
st.write(f"Predicted Class: {'Churn' if pred == 1 else 'Retain'}")

if prob < 0.30:
    st.success("Low Risk")
elif prob < 0.70:
    st.warning("Medium Risk")
else:
    st.error("High Risk")

st.subheader("Feature Importance")
st.bar_chart(feature_importance.set_index("Feature").head(10))

st.subheader("What-if Analysis")
st.write("Change the inputs in the sidebar to see how churn probability changes.")
