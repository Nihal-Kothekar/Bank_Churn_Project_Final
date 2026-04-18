from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Bank Customer Churn Dashboard", layout="wide")

DATA_PATH = Path("European_Bank.csv")
MODEL_PATH = Path("outputs/bank_churn_model.joblib")
FEATURE_PATH = Path("outputs/feature_importance.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def add_features(row_df: pd.DataFrame) -> pd.DataFrame:
    row_df = row_df.copy()
    row_df["BalanceSalaryRatio"] = row_df["Balance"] / (row_df["EstimatedSalary"] + 1)
    row_df["ProductDensity"] = row_df["NumOfProducts"] / (row_df["Tenure"] + 1)
    row_df["EngagementProductInteraction"] = row_df["IsActiveMember"] * row_df["NumOfProducts"]
    row_df["AgeTenureInteraction"] = row_df["Age"] * row_df["Tenure"]
    row_df["TenureByAge"] = row_df["Tenure"] / (row_df["Age"] + 1)
    return row_df


def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "High Risk"
    if prob >= 0.40:
        return "Medium Risk"
    return "Low Risk"


st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")
st.caption("Customer risk calculator, what-if analysis, and explainability dashboard")

if not DATA_PATH.exists():
    st.error("European_Bank.csv not found. Keep the CSV in the same folder as app.py.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run train_churn_model.py first.")
    st.stop()

try:
    df = load_data()
    model = load_model()
except Exception as exc:
    st.exception(exc)
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("Customer churn risk calculator")
    geography = st.selectbox("Geography", sorted(df["Geography"].dropna().unique().tolist()))
    gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique().tolist()))
    age = st.slider("Age", 18, 92, 40)
    credit_score = st.slider("Credit Score", 300, 900, 650)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    salary = st.number_input("Estimated Salary", min_value=1000.0, value=75000.0, step=1000.0)

    input_df = pd.DataFrame(
        [{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
        }]
    )
    input_df = add_features(input_df)

    proba = float(model.predict_proba(input_df)[0][1])
    prediction = int(model.predict(input_df)[0])

    st.metric("Churn Probability", f"{proba:.2%}")
    st.metric("Predicted Class", "Churn" if prediction == 1 else "Retained")
    st.metric("Risk Band", risk_label(proba))

with right:
    st.subheader("Probability visualization")
    chart_df = pd.DataFrame(
        {
            "Outcome": ["Retained", "Churn"],
            "Probability": [1 - proba, proba],
        }
    )
    fig = px.bar(chart_df, x="Outcome", y="Probability", text="Probability")
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("What-if scenario simulator")
    scenario_active = st.toggle("Switch customer to active member", value=bool(is_active))
    scenario_products = st.slider("Test a new product count", 1, 4, int(num_products), key="scenario_products")

    scenario_df = pd.DataFrame(
        [{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": scenario_products,
            "HasCrCard": has_card,
            "IsActiveMember": int(scenario_active),
            "EstimatedSalary": salary,
        }]
    )
    scenario_df = add_features(scenario_df)
    scenario_proba = float(model.predict_proba(scenario_df)[0][1])

    st.metric("Original Probability", f"{proba:.2%}")
    st.metric("Scenario Probability", f"{scenario_proba:.2%}", delta=f"{scenario_proba - proba:.2%}")

with col2:
    st.subheader("Feature importance dashboard")
    if FEATURE_PATH.exists():
        fi = pd.read_csv(FEATURE_PATH).head(10)
        fig2 = px.bar(fi.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run train_churn_model.py to generate feature_importance.csv.")

st.divider()
st.subheader("Dataset snapshot")
st.dataframe(df.head(10), use_container_width=True)
