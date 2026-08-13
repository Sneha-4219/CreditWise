# 💳 CreditWise — Loan Approval Prediction System

<p align="center">

A machine learning-based loan approval prediction system built with **Python, Scikit-learn, and Streamlit**.

Predict whether a loan application is likely to be **Approved** or **Rejected** based on applicant and loan-related information.

<br>

<a href="https://creditwise-sneha.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20Live%20Demo-CreditWise-success?style=for-the-badge" alt="Live Demo">
</a>

<a href="https://github.com/Sneha-4219/CreditWise">
  <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub Repository">
</a>

</p>

> ⚠️ **Disclaimer:** This project is intended for educational and demonstration purposes only and should not be used for real-world lending decisions.

---

## 🌐 Live Demo

🚀 **Try CreditWise:**  
https://creditwise-sneha.streamlit.app/

The application provides an interactive interface where users can enter applicant and loan details and receive a predicted loan approval outcome.

---

## 📌 Project Overview

CreditWise takes applicant information such as income, credit score, existing loans, savings, employment status, loan amount, loan term, and other applicant and loan-related features.

The data is preprocessed and transformed before being passed to the trained machine learning model.

The application then displays the predicted loan approval status through an interactive web interface built using Streamlit.

---

## 🚀 Features

- 📊 Loan approval prediction using Machine Learning
- 🤖 Gaussian Naive Bayes classification
- 🔄 Feature scaling using `StandardScaler`
- 🔢 Categorical feature encoding using `OneHotEncoder`
- 🏷️ Education Level encoding using `LabelEncoder`
- 📐 Feature engineering
- 🌐 Interactive Streamlit web application
- 💾 Saved trained model and preprocessing objects
- 📈 Model performance metrics displayed in the application

---

## 🧠 Machine Learning Pipeline

The project follows this workflow:

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Gaussian Naive Bayes
     ↓
Loan Approval Prediction
     ↓
Streamlit Web Application
```

---

## 📊 Model Performance

The trained Gaussian Naive Bayes model achieved the following results:

| Metric | Score |
|---|---|
| Accuracy | 86.00% |
| Precision | 81.13% |
| Recall | 70.49% |
| F1 Score | 75.44% |

These metrics are based on the model evaluation performed during the development of the project.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 📁 Project Structure

```text
CreditWise/
│
├── .gitignore
├── README.md
├── app.py
├── CreditWise_Loan_System.ipynb
├── loan_approval_data.csv
├── feature_columns.pkl
├── loan_model.pkl
├── onehot_encoder.pkl
├── scaler.pkl
└── requirements.txt
```

### File Description

| File | Description |
|---|---|
| app.py | Streamlit application |
| CreditWise_Loan_System.ipynb | Model development and experimentation |
| loan_approval_data.csv | Loan approval dataset |
| loan_model.pkl | Trained Gaussian Naive Bayes model |
| scaler.pkl | Saved feature scaler |
| onehot_encoder.pkl | Saved categorical encoder |
| feature_columns.pkl | Saved model feature column structure |
| requirements.txt | Python dependencies |
| .gitignore | Files excluded from Git |
| README.md | Project documentation |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sneha-4219/CreditWise.git
```

### 2. Navigate to the project directory
```bash
cd CreditWise
```

### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Application

The CreditWise application allows users to enter applicant and loan details such as:

- Applicant Income
- Coapplicant Income
- Age
- Dependents
- Existing Loans
- Savings
- Collateral Value
- Loan Amount
- Loan Term
- DTI Ratio
- Credit Score
- Education Level
- Employment Status
- Marital Status
- Loan Purpose
- Property Area
- Gender
- Employer Category

After entering the details, the model predicts whether the loan is likely to be Approved or Rejected.

---

## 🔍 Prediction Workflow

The application processes new applicant information using the same preprocessing approach used during model development.

```text
Applicant Details
       ↓
Categorical Encoding
       ↓
Feature Engineering
       ↓
Feature Scaling
       ↓
Trained Gaussian Naive Bayes Model
       ↓
Loan Approval Prediction
```

The saved preprocessing objects ensure that new applicant data is transformed consistently before being passed to the trained model.

---

## 📈 Model Features

The trained model uses 28 features, including original, encoded, and engineered features.

The feature set includes:

- Applicant_Income
- Coapplicant_Income
- Age
- Dependents
- Existing_Loans
- Savings
- Collateral_Value
- Loan_Amount
- Loan_Term
- Education_Level
- Employment_Status_Salaried
- Employment_Status_Self-employed
- Employment_Status_Unemployed
- Marital_Status_Single
- Loan_Purpose_Car
- Loan_Purpose_Education
- Loan_Purpose_Home
- Loan_Purpose_Personal
- Property_Area_Semiurban
- Property_Area_Urban
- Gender_Male
- Employer_Category_Government
- Employer_Category_MNC
- Employer_Category_Private
- Employer_Category_Unemployed
- DTI_Ratio_sq
- Credit_Score_sq
- Applicant_Income_log

---

## 🔧 Preprocessing

The project uses different preprocessing techniques for numerical and categorical information.

### Categorical Features

Categorical variables are transformed using:

- OneHotEncoder
- LabelEncoder

The OneHotEncoder is used for categorical variables such as:

- Employment_Status
- Marital_Status
- Loan_Purpose
- Property_Area
- Gender
- Employer_Category

Education Level is encoded separately using a LabelEncoder.

### Numerical Features

Numerical features are scaled using a saved StandardScaler object before prediction.

### Feature Engineering

Additional features are created to help represent relationships in the data, including:

- DTI_Ratio_sq
- Credit_Score_sq
- Applicant_Income_log

---

## 💾 Saved Model Components

The project saves the trained model and preprocessing components using Joblib.

- loan_model.pkl
- scaler.pkl
- onehot_encoder.pkl
- feature_columns.pkl

These files allow the Streamlit application to load the trained model and process new inputs without retraining the model every time.

---

## 🧪 Example Prediction

A user enters applicant information through the Streamlit interface and clicks:

**🔍 Predict Loan Approval**

The application then displays one of the following results:

### 🎉 Loan Approved!

or

### ❌ Loan Rejected

---

## 👩‍💻 Author

**Sneha Sharma**

B.Tech CSE — AI/ML

GitHub: [Sneha-4219](https://github.com/Sneha-4219)

---

## ⭐ Acknowledgement

This project was developed as a machine learning project to demonstrate the practical implementation of a classification model and its deployment through an interactive web application.