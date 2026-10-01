<div align="center">
  <img src="https://meobserver.news/wp-content/uploads/2025/03/20250203T101527926.png" alt="DEBI Logo" width="180" />

  <h1>📊 Customer 360 Intelligence: End-to-End ML Pipeline</h1>

  <strong>Digital Egypt Builders Initiative (DEBI) — Machine Learning Engineer Track</strong><br>
  <strong>Instructor:</strong> Eng. <a href="https://github.com/mhemaly">Mahmoud El-Sayed</a> (<a href="https://github.com/mhemaly">@mhemaly</a>)<br>
  <strong>Academic Year:</strong> 2026<br><br>

  <img src="https://img.shields.io/badge/Python-3.7.16-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.0.2-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Pandas-1.3.5-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-1.21.6-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Jupyter-1.0.0-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</div>

---

## 📋 Project Overview

A comprehensive Customer 360 analytics solution for *NileConnect*, a fictional telecom and digital-services provider. This project executes a complete, end-to-end Machine Learning workflow—from data understanding and rigorous preprocessing to model evaluation and business interpretation.

The platform analyzes **3,000 realistic customer records** to extract actionable business intelligence regarding customer retention, expected lifetime value, and behavioral segmentation.

> **Key Business Objectives:**
> 1. **Churn Prediction:** Identify customers likely to leave (Classification).
> 2. **Revenue Forecasting:** Estimate the 12-month future revenue per customer (Regression).
> 3. **Customer Segmentation:** Discover natural customer groups for targeted marketing (Clustering).
> 4. **Anomaly Detection:** Flag customers with highly unusual behavior for further investigation (Isolation Forest).

---

## 🎓 Acknowledgments

This capstone project was completed as part of the **Digital Egypt Builders Initiative (DEBI)**.

I would like to express my sincere gratitude to **Eng. [Mahmoud El-Sayed](https://github.com/mhemaly)** ([@mhemaly](https://github.com/mhemaly)) for his exceptional guidance, insightful feedback, and dedication to teaching us the core principles of Machine Learning and business-oriented AI solutions throughout this workshop.

---

## 🧩 ML Pipeline Architecture Deep Dive

The project strictly follows an End-to-End ML Pipeline architecture to ensure reproducibility and prevent target leakage:

```text
Raw Data (customer_360_ml_workshop.csv)
             ↓
     Data Understanding & EDA       ← Identifying patterns & leakage
             ↓
    Preprocessing Pipeline          ← Median/Most-Frequent Imputation, Standard Scaling, OHE
             ↓
 -----------------------------------------
 |                   |                   |
Classification   Regression         Clustering & PCA
 (Churn)         (Revenue)        (Segmentation & Anomalies)
 |                   |                   |
 -----------------------------------------
             ↓
    Model Evaluation & Tuning       ← GridSearchCV, Cross-Validation
             ↓
    Customer 360 Integration        ← Final actionable business table
             ↓
    outputs/customer_360_predictions.csv
```

---

## ✨ Core Modules & Key Features

### 🔍 Part A & B: EDA & Data Preprocessing
- **Identifier Removal Logic** — Strict exclusion of `CustomerID` to prevent model memorization.
- **Target Leakage Prevention** — Ensuring `Future12MRevenueEGP` is excluded when predicting current churn.
- **Robust Scikit-Learn Pipeline** — A unified preprocessing transformer handling Imputation, Standardization, and One-Hot Encoding (`handle_unknown="ignore"`).

### 🎯 Part C: Churn Classification (Supervised)
- **Algorithm Comparison** — Evaluates Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, and Gradient Boosting.
- **Hyperparameter Tuning** — Utilizes `GridSearchCV` to optimize the Random Forest model on a stratified train/test split.
- **Evaluation Metrics** — Generates Confusion Matrices, ROC Curves, Accuracy, Precision, Recall, F1, and ROC-AUC.

### 💰 Part D: Revenue Forecasting (Regression)
- **Algorithm Comparison** — Evaluates Linear Regression, Ridge, Lasso, and Tree-based Regressors.
- **Performance Evaluation** — Strictly scored using MAE, RMSE, and R². Plots Actual vs. Predicted Revenue to validate model reliability.

### 📊 Part E & F: Segmentation, PCA & Anomaly Detection (Unsupervised)
- **Clustering Algorithms** — Applies K-Means, Agglomerative Clustering, and DBSCAN on behavioral variables. Optimal K is algorithmically justified using Silhouette Scores and Inertia.
- **Dimensionality Reduction** — Utilizes PCA (2 Components) for 2D visualization of high-dimensional segmentation.
- **Isolation Forest** — Detects highly unusual customers without automatically mislabeling them as "fraud."

### ⚙️ Part G: Customer 360 Integration
- **Final Output Generation** — A synthesized CSV containing `CustomerID`, churn probability, predicted revenue, segment, and anomaly flag.
- **Retention-Priority Engine** — A custom logic rule assigning retention priority based on a matrix of high churn probability and high forecasted revenue.

---

## 🔑 Key Design Decisions

| Decision | Description |
|----------|-------------|
| **Strict Data Segregation** | Preprocessing is explicitly fitted after the train/test split to completely eliminate data leakage. |
| **Business-First Metrics** | While accuracy is measured, Recall and Precision are heavily weighted for Churn Prediction depending on the cost of false positives vs. false negatives. |
| **Pipeline Reusability** | The scikit-learn pipeline guarantees that new, raw customer data can be processed for inference without manual intervention. |
| **Reproducibility** | Seed `random_state=42` is enforced across all applicable algorithms. |

---

## 🚀 Getting Started & Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/AbdullahMAhmoud/ML_Workshop_Customer360.git
cd ML_Workshop_Customer360

# 2. Create a Python 3.7 environment (using Conda)
conda create -n ml-workshop python=3.7.16 -y
conda activate ml-workshop
pip install -r requirements.txt

# 3. Verify the environment
python check_environment.py

# 4. Start working
python src/workshop_tasks.py
# OR
jupyter notebook
```

---

## 📁 Repository Structure

```text
ML_Workshop_Customer360/
│
├── data/
│   └── customer_360_ml_workshop.csv      ← Synthetic dataset (3,000 records)
├── src/
│   └── workshop_tasks.py                 ← Starter script
├── notebooks/                            ← Jupyter notebooks for EDA and ML phases
├── outputs/
│   └── customer_360_predictions.csv      ← Final integrated predictions
├── README.md
├── requirements.txt
└── check_environment.py
```
