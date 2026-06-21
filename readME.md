# Credit Risk Prediction System

## Project Overview

### Project Title

**Credit Risk Prediction System – End-to-End Machine Learning Model with Deployment**

### Objective

The objective of this project is to develop a Machine Learning-based system that predicts whether a loan applicant is a **High-Risk** or **Low-Risk** borrower. The solution helps financial institutions reduce loan defaults and improve loan approval decisions through data-driven risk assessment.

---

# Dataset Description

The dataset contains financial, demographic, and behavioral information of loan applicants.

## Input Features

* Income
* Employment Type
* Loan Amount
* Loan Duration
* Credit Score
* Age
* Dependents
* Number of Previous Defaults
* Collateral Information
* Marital Status
* Loan Repayment History

## Target Variable

**Risk**

* 0 → Low Risk
* 1 → High Risk

---

# Business Problem

Loan defaults can result in significant financial losses for banks and lending institutions. The goal of this project is to identify risky borrowers before loan approval and support better lending decisions.

---

# Data Preprocessing

The following preprocessing steps were performed:

* Missing value handling using suitable imputation techniques.
* Removal of duplicate records.
* Data quality assessment and validation.
* Encoding of categorical features using One-Hot Encoding.
* Standardization of numerical variables using StandardScaler.
* Feature transformation for machine learning compatibility.

---

# Feature Engineering

To improve model performance, additional features were created.

## Debt-to-Income Ratio

Measures the applicant's debt burden relative to income.

Debt-to-Income Ratio = Loan Amount / Income

## Total Obligation Ratio

Measures overall financial obligations relative to income.

Total Obligation Ratio = Total Debt / Income

## Encoded Features

Categorical variables such as:

* Employment Type
* Marital Status
* Loan Repayment History
* Collateral Information

were converted into numerical representations using One-Hot Encoding.

---

# Exploratory Data Analysis (EDA)

EDA was conducted to understand the dataset structure and identify patterns associated with loan risk.

## Visualizations Performed

### Risk Distribution

* Count Plot
* Pie Chart

### Income Analysis

* Histogram
* Distribution Plot

### Loan Amount Analysis

* Histogram

### Credit Score Analysis

* Box Plot
* Distribution Plot

### Age Analysis

* Histogram
* Box Plot

### Correlation Analysis

* Correlation Heatmap

### Categorical Feature Analysis

* Employment Type vs Risk
* Marital Status vs Risk
* Repayment History vs Risk
* Collateral Information vs Risk

## Key Insights

* Applicants with lower credit scores showed higher default risk.
* Previous loan defaults strongly influenced risk prediction.
* Debt-to-Income Ratio emerged as an important predictive feature.
* Repayment history significantly impacted loan risk classification.
* Some categorical variables showed strong associations with borrower risk.

---

# Machine Learning Model Development

Multiple classification algorithms were considered and compared.

## Models Evaluated

### Logistic Regression

* Simple and interpretable baseline model.

### Decision Tree Classifier

* Captures non-linear relationships.
* Easy to interpret.

### Random Forest Classifier

* Reduces overfitting.
* Provides robust performance.

### Gradient Boosting Classifier

* Improves prediction accuracy through boosting.

### XGBoost Classifier

* High-performance gradient boosting algorithm.
* Excellent for structured tabular data.

---

# Model Training

## Train-Test Split

Dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

The training set was used to learn patterns, while the testing set was used for unbiased performance evaluation.

---

# Model Evaluation Metrics

The models were evaluated using the following metrics:

## Accuracy

Measures overall prediction correctness.

## Precision

Measures how many predicted risky applicants were actually risky.

## Recall

Measures how many actual risky applicants were correctly identified.

## F1 Score

Provides a balance between Precision and Recall.

## ROC-AUC Score

Measures overall classification capability across different decision thresholds.

## Confusion Matrix

Provides detailed classification performance analysis.

---

# Best Model Selection

The final model was selected based on:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Generalization Performance

After comparison, **XGBoost** was selected as the final deployment model due to its superior predictive performance.

---

# Model Deployment

The final model was deployed using **Streamlit**.

## Application Features

### User Input Form

Users can enter:

* Income
* Loan Amount
* Credit Score
* Age
* Loan Duration
* Dependents
* Previous Defaults
* Employment Type
* Marital Status
* Repayment History
* Collateral Information

### Real-Time Prediction

The application predicts:

* High Risk
* Low Risk

### Probability Score

Displays confidence levels for predictions.

### Feature Importance Visualization

Shows the most influential features contributing to model predictions.

### Downloadable Report

Allows users to download a prediction report for future reference.

---

# Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn
* Plotly

## Machine Learning

* Scikit-Learn
* XGBoost

## Model Serialization

* Joblib

## Deployment

* Streamlit

---

# Project Workflow

1. Data Understanding and Requirement Analysis
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Exploratory Data Analysis
5. Model Training
6. Model Evaluation
7. Model Selection
8. Streamlit Deployment
9. Final Documentation

---

# Learning Outcomes

Through this project, the following skills were developed:

* Data Cleaning and Preprocessing
* Feature Engineering
* Exploratory Data Analysis
* Classification Modeling
* Hyperparameter Tuning
* Model Evaluation
* Credit Risk Analytics
* Streamlit Deployment
* End-to-End Machine Learning Workflow
* Real-World Problem Solving

---

# Conclusion

The Credit Risk Prediction System successfully demonstrates a complete Machine Learning pipeline for financial risk assessment. By utilizing financial, demographic, and behavioral features, the model accurately predicts whether a borrower is likely to be a High-Risk or Low-Risk applicant. The deployed Streamlit application provides real-time predictions, risk probabilities, feature importance insights, and downloadable reports, making it a practical solution for modern credit risk management.

