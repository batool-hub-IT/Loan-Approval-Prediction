# Loan Approval Prediction

## Project Overview
Loan Approval Prediction is a Machine Learning based application that predicts whether a loan application is likely to be approved or rejected.

The project uses applicant financial and personal information as input and applies a Logistic Regression model to generate a loan approval prediction along with an approval probability.

The trained Machine Learning model is integrated with a Flask web application through a simple and user-friendly interface.

## Problem Statement
Loan approval decisions depend on multiple factors such as income, credit score, loan amount, loan term, employment status, and asset values.

The goal of this project is to develop a Machine Learning system that can analyze these factors and predict the loan approval status of an applicant.

## Dataset
The project uses a Loan Approval Prediction dataset containing 4,269 records.

### Features
- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

### Target Variable
- Approved
- Rejected

## Data Preprocessing
The following preprocessing steps were performed:

1. Removed the unnecessary loan_id column from the input features.
2. Cleaned leading and trailing spaces from column names and categorical values.
3. Encoded categorical variables: Graduate = 1, Not Graduate = 0, Yes = 1, No = 0.
4. Encoded the target variable: Approved = 1, Rejected = 0.
5. Split the dataset into training and testing sets.
6. Used an 80/20 train-test split with stratification.
7. Applied StandardScaler to the input features.

### Dataset Split
- Training samples: 3,415
- Testing samples: 854

## Exploratory Data Analysis
Exploratory Data Analysis was performed to understand the dataset, identify patterns, and investigate relationships between applicant characteristics and loan approval.

Important variables investigated included:
- CIBIL Score
- Annual Income
- Loan Amount
- Loan Term
- Number of Dependents
- Asset Values
- Education
- Self Employment

## Machine Learning Model
### Logistic Regression
Logistic Regression was used because the target variable contains two classes: Approved and Rejected.

The model was trained using the scaled training data.

## Model Evaluation
The model was evaluated using the test dataset.

### Accuracy
**91.33%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Rejected | 0.90 | 0.87 | 0.88 |
| Approved | 0.92 | 0.94 | 0.93 |

### Confusion Matrix

| | Predicted Rejected | Predicted Approved |
|---|---:|---:|
| Actual Rejected | 280 | 43 |
| Actual Approved | 31 | 500 |

The model correctly classified 780 out of 854 test applications.

## Web Application
The trained Machine Learning model was integrated into a Flask web application.

### Application Flow
1. User enters loan information.
2. Flask receives the input.
3. Input data is scaled using the trained scaler.
4. Logistic Regression generates a prediction.
5. Approval probability is calculated.
6. The result is displayed in the web interface.

The application provides:
- Loan approval prediction
- Approval probability
- Simple web-based interface

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS
- JavaScript
- Google Colab
- GitHub

## Project Structure

Loan-Approval-Prediction/
├── templates/
│   └── index.html
├── app.py
├── loan_approval_model.pkl
├── loan_approval_scaler.pkl
├── loan_approval_dataset.csv
├── requirements.txt
└── README.md

## Limitations
- The model is trained on a specific dataset, so performance may differ on other datasets or real-world applications.
- Predictions are Machine Learning estimates and should not be treated as final financial decisions.
- The dataset may not contain every factor used in real-world loan assessment.
- The application does not replace professional financial or credit assessment.

## Future Improvements
- Explainable AI for showing factors behind predictions
- Loan affordability analysis
- What-if loan simulation
- Suspicious application/anomaly detection
- Comparison of multiple Machine Learning models
- Cloud deployment
- Improved input validation

## Conclusion
This project demonstrates an end-to-end Machine Learning workflow, from data preprocessing and exploratory analysis to model training, evaluation, and integration with a Flask web application.

The Logistic Regression model achieved 91.33% accuracy on the test dataset and was successfully integrated into a working loan approval prediction system.