import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("🩺 Diabetes Risk Prediction System")
st.write("Predict diabetes risk using clinical patient data.")

st.divider()

# -----------------------------
# Input Section
# -----------------------------

st.subheader("Enter Patient Information")

preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=250, value=120)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=5, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=15, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=18, max_value=100, value=30)

st.divider()

# -----------------------------
# Risk Category Function
# -----------------------------

def risk_category(prob):
    if prob < 0.3:
        return "Low Risk 🟢"
    elif prob < 0.6:
        return "Moderate Risk 🟡"
    else:
        return "High Risk 🔴"

# -----------------------------
# Prediction Section
# -----------------------------

if st.button("Predict Risk"):

    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    category = risk_category(prob)

    st.subheader("Prediction Result")
    st.write("Risk Probability:", f"{round(prob * 100, 1)}%")
    st.write("Risk Category:", category)

    # Color-coded feedback
    if prob >= 0.6:
        st.error("High probability of diabetes detected. Clinical consultation recommended.")
    elif prob >= 0.3:
        st.warning("Moderate risk detected. Lifestyle monitoring advised.")
    else:
        st.success("Low diabetes risk predicted.")

    # -----------------------------
    # Patient-Specific Feature Contribution
    # -----------------------------
    st.divider()
    st.subheader("Top Contributing Factors (Patient-Specific)")

    feature_names = [
        "Pregnancies","Glucose","BloodPressure",
        "SkinThickness","Insulin","BMI",
        "DiabetesPedigreeFunction","Age"
    ]

    coefficients = model.coef_[0]
    patient_values = input_scaled[0]

    # Approximate contribution per feature
    contributions = coefficients * patient_values

    sorted_features = sorted(
        zip(feature_names, contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for name, contrib in sorted_features[:3]:
        direction = "↑ Increased Risk" if contrib > 0 else "↓ Reduced Risk"
        st.write(f"{name}: {round(contrib, 3)} ({direction})")

st.divider()

# -----------------------------
# Model Information Section
# -----------------------------

st.subheader("Model Information")

st.write("Model Used: Logistic Regression")
st.write("Dataset: Pima Indians Diabetes Dataset")
st.write("Class Weight: Balanced")
st.write("Features Standardized using StandardScaler")

st.caption("⚠️ This system is for educational purposes only and does not replace professional medical advice.")
st.caption("Model confidence increases when clinical values are within realistic ranges.")
