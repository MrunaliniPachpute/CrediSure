# 🏦 CrediSure Loan Approval System

An end-to-end Machine Learning project that predicts whether a loan application should be approved or rejected based on applicant financial and personal details.

---

## 🚀 Project Overview

Financial institutions often face challenges in:
- ❌ Rejecting eligible customers 
- ❌ Approving high-risk applicants

This project builds an intelligent system that:
- Analyzes applicant data
- Learns patterns from historical records
- Predicts loan approval decisions with high precision

---

## 📊 Dataset Features

The dataset includes:
- Applicant Income, Coapplicant Income
- Credit Score, DTI Ratio
- Savings, Loan Amount, Loan Term
- Employment Status, Education Level
- Property Area, Loan Purpose, etc.

Target:
- **Loan_Approved (1 = Approved, 0 = Rejected)**

---

## 🧠 Machine Learning Approach

### ✔ Data Preprocessing
- Handled missing values using:
  - Mean (numerical)
  - Mode (categorical)
- Label Encoding + One-Hot Encoding

### ✔ Feature Engineering
- Squared Features:
  - `DTI_Ratio²`
  - `Credit_Score²`
- Log Transformation:
  - `log(Applicant_Income)`

### ✔ Models Implemented
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes

---

## 🎯 Evaluation Strategy

Since approving high-risk customers is costly:

👉 **Precision was prioritized over recall**

### 🏆 Best Model:
**Naive Bayes**

- Precision: ~0.81
- Accuracy: ~0.86

---

## 💻 Deployment

The model is deployed using **Streamlit** for real-time predictions. \n
Check Out : https://credisure-guytm4ja8llfpk5rke68qs.streamlit.app/

### Features:
- User-friendly interface
- Input applicant details
- Instant loan approval prediction
- Confidence score display

---

## 📸 Demo

<img width="500" height="600" alt="Screenshot (1645)" src="https://github.com/user-attachments/assets/0216a022-6a97-43f5-a16b-96d57f72c37e" />
<img width="500" height="600" alt="Screenshot (1644)" src="https://github.com/user-attachments/assets/f7fdb2f6-41ee-4b6e-9644-118cf2b3fc15" />

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
