# CreditWise-Loan-Approval-System
**ML-powered loan decision engine** for SecureTrust Bank · Predicts loan approval using applicant financial & personal data.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2E7D32?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

---

## 📌 Problem Statement

SecureTrust Bank processes hundreds of loan applications daily via branch and online channels. The existing **manual review process** is slow, inconsistent, and prone to human bias — resulting in:

- ❌ Good applicants being rejected (lost revenue)
- ❌ High-risk applicants being approved (financial loss)

**CreditWise** solves this by automating the preliminary decision using a trained ML classifier, allowing loan officers to focus only on edge cases.

---

## 📂 Project Structure

```
CreditWise-Loan-System/
│
├── credit_wise.ipynb          # Main notebook (EDA → Training → Evaluation)
├── loan_approval_data.csv     # Dataset (1,000 applicant records)
├── CreditWise_Loan_System.pdf # Project overview document
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 📊 Dataset

| Property        | Detail                        |
|----------------|-------------------------------|
| Rows           | 1,000 applicant records       |
| Features       | 19 input columns              |
| Target         | `Loan_Approved` (0 = Rejected, 1 = Approved) |
| Feature Types  | Numeric + Categorical         |

### Key Features Used
| Feature | Description |
|---|---|
| `Credit_Score` | Bureau credit score (strongest predictor) |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Applicant_Income` | Monthly income |
| `Savings` | Current savings balance |
| `Collateral_Value` | Value of collateral offered |
| `Employment_Status` | Salaried / Self-Employed / Business |
| `Loan_Amount` | Requested loan amount |
| ... | *(19 features total — see PDF for full list)* |

---

## 🔬 ML Pipeline

```
Raw Data
   │
   ├─ 1. Missing Value Handling   → Mean (numeric) / Mode (categorical)
   ├─ 2. EDA                      → Distribution, boxplots, heatmap
   ├─ 3. Encoding                 → Label Encoding + One-Hot Encoding
   ├─ 4. Feature Engineering      → DTI² and Credit_Score² polynomial features
   ├─ 5. Train / Test Split       → 80:20, random_state=42
   ├─ 6. Feature Scaling          → StandardScaler (fit on train only)
   └─ 7. Model Training           → Logistic Regression · KNN · Naive Bayes
```

---

## 🤖 Models & Results

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| K-Nearest Neighbours | — | — | — | — |
| **Naive Bayes ✅** | **Best** | — | — | — |

> ✅ **Naive Bayes** achieved the highest **Precision** after feature engineering and was selected as the final model.

*(Run the notebook to see exact metric values from your environment)*

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML models, preprocessing, metrics |
| Seaborn / Matplotlib | Data visualisation |
| Jupyter Notebook | Interactive development |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Abhinna Singh/CreditWise-Loan-System.git
cd CreditWise-Loan-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook credit_wise.ipynb
```

> Make sure `loan_approval_data.csv` is in the **same directory** as the notebook before running.

---

## 📋 requirements.txt (copy this into your repo)

```
pandas
numpy
scikit-learn
seaborn
matplotlib
jupyter
```

---

## 📈 Key Insights

- 🔑 **Credit Score** is the single strongest predictor of loan approval
- 📉 Higher **DTI Ratio** strongly correlates with rejection
- 💰 **Savings balance** positively impacts approval probability
- ⚙️ Polynomial features (DTI² and Credit_Score²) improved model precision
- ✅ **Naive Bayes** outperformed Logistic Regression and KNN on Precision

---

## 📄 Documentation

See [`CreditWise_Loan_System.pdf`](./CreditWise_Loan_System.pdf) for the full problem statement, dataset dictionary, and pipeline overview.

---

## 🙋 Author

**Abhinna Singh**
- 💼 [LinkedIn](https://www.linkedin.com/in/abhinna-singh-940b6b402 )
- 🐙 [GitHub](https://github.com/abhinna-singh)
- 📧 ry5777529@gmail.com

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

*Built as part of a Machine Learning Engineering project for SecureTrust Bank's CreditWise system.*
