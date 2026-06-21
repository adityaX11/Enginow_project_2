import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("loan_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.big-font {
    font-size:40px !important;
    font-weight:bold;
    text-align:center;
}

.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<p class='big-font'>🏦 Credit Risk Prediction System</p>",
    unsafe_allow_html=True
)

st.markdown(
"""
Predict whether a loan applicant is likely to be:

- 🟢 Low Risk
- 🔴 High Risk

using Machine Learning (XGBoost).
"""
)

st.divider()

# =====================================================
# SIDEBAR INPUT FORM
# =====================================================

st.sidebar.header("Applicant Information")

income = st.sidebar.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    700
)

age = st.sidebar.slider(
    "Age",
    18,
    80,
    30
)

loan_duration = st.sidebar.slider(
    "Loan Duration (Months)",
    6,
    360,
    60
)

previous_defaults = st.sidebar.slider(
    "Previous Defaults",
    0,
    10,
    0
)

dependents = st.sidebar.slider(
    "Dependents",
    0,
    10,
    1
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    [
        "Salaried",
        "Self-Employed",
        "Business",
        "Unemployed"
    ]
)

collateral_info = st.sidebar.selectbox(
    "Collateral Information",
    [
        "Yes",
        "No"
    ]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

repayment_history = st.sidebar.selectbox(
    "Loan Repayment History",
    [
        "Good",
        "Average",
        "Poor"
    ]
)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

debt_income_ratio = (
    loan_amount / income
    if income > 0 else 0
)

total_obligation_ratio = (
    loan_amount / income
    if income > 0 else 0
)

# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame({

    "Income":[income],
    "LoanAmount":[loan_amount],
    "CreditScore":[credit_score],
    "Age":[age],
    "LoanDuration":[loan_duration],
    "NumberPreviousDefaults":[previous_defaults],
    "Dependents":[dependents],
    "Debt_to_Income_Ratio":[debt_income_ratio],
    "Total_Obligation_Ratio":[total_obligation_ratio],

    # Example encoded values

    "EmploymentType_Self-Employed":[1 if employment_type=="Self-Employed" else 0],
    "EmploymentType_Salaried":[1 if employment_type=="Salaried" else 0],
    "EmploymentType_Business":[1 if employment_type=="Business" else 0],

    "CollateralInformation_Yes":[1 if collateral_info=="Yes" else 0],

    "MaritalStatus_Married":[1 if marital_status=="Married" else 0],
    "MaritalStatus_Single":[1 if marital_status=="Single" else 0],

    "LoanRepaymentHistory_Good":[1 if repayment_history=="Good" else 0],
    "LoanRepaymentHistory_Average":[1 if repayment_history=="Average" else 0]

})

# =====================================================
# MATCH TRAINING FEATURES
# =====================================================

for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_names]

# =====================================================
# PREDICT BUTTON
# =====================================================

predict_btn = st.button(
    "🔍 Predict Risk",
    use_container_width=True
)

# =====================================================
# PREDICTION
# =====================================================

if predict_btn:

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    low_risk_prob = probability[0]
    high_risk_prob = probability[1]

    st.divider()

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"🔴 HIGH RISK\n\nConfidence: {high_risk_prob:.2%}"
            )

        else:

            st.success(
                f"🟢 LOW RISK\n\nConfidence: {low_risk_prob:.2%}"
            )

        st.progress(float(max(probability)))

    with col2:

        st.metric(
            "Risk Probability",
            f"{high_risk_prob:.2%}"
        )

        st.metric(
            "Credit Score",
            credit_score
        )

    # ===============================================
    # APPLICANT SUMMARY
    # ===============================================

    st.subheader("Applicant Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Income",
            "Loan Amount",
            "Credit Score",
            "Age",
            "Previous Defaults"
        ],

        "Value":[
            income,
            loan_amount,
            credit_score,
            age,
            previous_defaults
        ]
    })

    st.dataframe(summary, use_container_width=True)

    # ===============================================
    # FEATURE IMPORTANCE
    # ===============================================

    st.subheader("Feature Importance")

    try:

        importance = pd.DataFrame({

            "Feature": feature_names,
            "Importance": model.feature_importances_

        })

        importance = (
            importance
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except:
        st.info(
            "Feature importance unavailable."
        )

    # ===============================================
    # DOWNLOAD REPORT
    # ===============================================

    report = f"""
Credit Risk Report

Prediction:
{'HIGH RISK' if prediction == 1 else 'LOW RISK'}

High Risk Probability:
{high_risk_prob:.2%}

Income:
{income}

Loan Amount:
{loan_amount}

Credit Score:
{credit_score}
"""

    st.download_button(
        "📄 Download Report",
        report,
        file_name="risk_report.txt"
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
### About Project

This system predicts loan default risk using
Machine Learning and financial indicators.

Model:
- XGBoost

Target:
- High Risk
- Low Risk
""")