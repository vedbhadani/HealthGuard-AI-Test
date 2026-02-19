# 🩺 Intelligent Patient Risk Assessment System (Milestone 1)

## 📌 Project Overview

This project implements a Machine Learning-based healthcare analytics system that predicts diabetes risk using structured clinical data.

The system analyzes patient health indicators and provides:

- Risk probability (%)
- Risk category (Low / Moderate / High)
- Top contributing clinical factors

This project fulfills **Milestone 1: ML-Based Patient Risk Assessment**.

---

## 🎯 Problem Statement

Diabetes is a chronic metabolic disorder affecting millions globally. Early risk detection is critical to prevent long-term complications such as cardiovascular disease, kidney failure, and nerve damage.

This system aims to:

- Predict diabetes risk using structured clinical features
- Provide interpretable risk explanations
- Support preliminary screening and preventive care

⚠️ This system is intended for educational purposes only and does not replace professional medical diagnosis.

---

## 📊 Dataset

**Dataset Used:** Pima Indians Diabetes Dataset  
Source: Kaggle / UCI Machine Learning Repository  

### Features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

### Target:
- Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## ⚙️ System Architecture (ML Pipeline)

```mermaid
flowchart TD

A[Patient Input - Streamlit UI]
B[Data Preprocessing<br>• Zero Handling<br>• Median Imputation<br>• StandardScaler]
C[Logistic Regression Model]
D[Probability Output]
E[Risk Categorization]
F[Feature Contribution Analysis]
G[UI Display]

A --> B
B --> C
C --> D
D --> E
C --> F
E --> G
F --> G
```

## 🚀 How to Run Locally

```bash
# 1️⃣ Clone the repository
git clone <your-repository-link>
cd diabetes-risk-ml

# 2️⃣ Create virtual environment
python3 -m venv .venv

# 3️⃣ Activate virtual environment

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 4️⃣ Install dependencies
pip install -r requirements.txt

# 5️⃣ Train the model
python train.py

# 6️⃣ Run the Streamlit application
streamlit run app.py

# 7️⃣ Open in browser
http://localhost:8501

