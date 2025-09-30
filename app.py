import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import random

# Required imports for sklearn model unpickling
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

# --- Load Model ---
@st.cache_resource
def load_model():
    deploy_obj = joblib.load("ahd_model_C_hybrid_fixed.pkl")
    model = deploy_obj['model']
    features = deploy_obj.get('feature_names', None)
    return model, features

model, feature_names = load_model()

# --- Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Prediction Tool", "Dashboard", "Guideline Chatbot"])

# =====================================================
# PAGE 1: Prediction Tool
# =====================================================
if page == "Prediction Tool":
    st.title("🧑‍⚕️ HIV Advanced Disease Risk Predictor")

    # User Inputs
    age = st.number_input("Age (years)", 1, 100, 35)
    weight = st.number_input("Weight (kg)", 20, 150, 60)
    height = st.number_input("Height (cm)", 100, 210, 165)
    cd4 = st.number_input("Latest CD4 (cells/µL)", 0, 1500, 350)
    vl = st.number_input("Last Viral Load (copies/mL)", 0, 1000000, 500)
    months_art = st.number_input("Months on ART", 0, 300, 36)
    sex = st.selectbox("Sex", ["M", "F"])
    who_stage = st.selectbox("Last WHO Stage", ["1", "2", "3", "4"])

    # Feature Engineering
    bmi = weight / ((height/100)**2)

    stage_flags = {
        "Last_WHO_Stage_2": 1 if who_stage == "2" else 0,
        "Last_WHO_Stage_3": 1 if who_stage == "3" else 0,
        "Last_WHO_Stage_4": 1 if who_stage == "4" else 0,
    }

    vl_suppressed = 1 if vl < 1000 else 0

    cd4_risk = {
        "cd4_risk_Severe": 1 if cd4 < 200 else 0,
        "cd4_risk_Moderate": 1 if 200 <= cd4 < 350 else 0,
        "cd4_risk_Normal": 1 if cd4 >= 350 else 0,
    }

    input_data = {
        "Age at reporting": age,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Latest CD4 Result": cd4,
        "CD4_Missing": 0,
        "Last VL Result": vl,
        "VL_Suppressed": vl_suppressed,
        "VL_Missing": 0,
        "Months of Prescription": months_art,
        "Sex_M": 1 if sex == "M" else 0,
        **stage_flags,
        **cd4_risk,
        "Active_in_PMTCT_Missing": 0,
        "Cacx_Screening_Missing": 0,
        "Refill_Date_Missing": 0,
    }

    X = pd.DataFrame([input_data])

    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)

    if st.button("Predict Risk"):
        prob = model.predict_proba(X)[0][1] * 100
        st.metric("Predicted Risk of Advanced HIV Disease", f"{prob:.2f}%")
        if prob > 50:
            st.warning("⚠️ High risk – consider urgent evaluation.")
        else:
            st.success("✅ Lower risk – continue monitoring as per guidelines.")

# =====================================================
# PAGE 2: Dashboard
# =====================================================
elif page == "Dashboard":
    st.title("📊 HIV & NCD Dashboard")

    # Simulated dataset (replace with real)
    np.random.seed(42)
    df = pd.DataFrame({
        "Age": np.random.randint(18, 70, 200),
        "CD4": np.random.randint(50, 1000, 200),
        "VL": np.random.randint(50, 50000, 200),
        "BMI": np.round(np.random.normal(23, 4, 200), 1)
    })

    st.write("Example simulated patient data (replace with real clinical dataset):")
    st.dataframe(df.head())

    # Plot CD4 distribution
    fig, ax = plt.subplots()
    ax.hist(df["CD4"], bins=20, color="skyblue", edgecolor="black")
    ax.set_title("CD4 Count Distribution")
    ax.set_xlabel("CD4 (cells/µL)")
    ax.set_ylabel("Number of Patients")
    st.pyplot(fig)

    # Plot BMI distribution
    fig2, ax2 = plt.subplots()
    ax2.hist(df["BMI"], bins=20, color="lightgreen", edgecolor="black")
    ax2.set_title("BMI Distribution")
    ax2.set_xlabel("BMI")
    ax2.set_ylabel("Number of Patients")
    st.pyplot(fig2)

# =====================================================
# PAGE 3: Guideline Chatbot
# =====================================================
elif page == "Guideline Chatbot":
    st.title("💬 HIV/NCD Guideline Chatbot")

    st.write("Ask me anything about HIV care, AHD, or NCDs. Responses are based on WHO guidelines.")

    # Mini knowledge base (could later be replaced by proper RAG)
    knowledge_base = {
        "what is advanced hiv disease": "WHO defines Advanced HIV Disease as CD4 < 200 cells/µL or WHO clinical stage 3 or 4.",
        "what are common ncds in hiv": "People living with HIV are at higher risk of hypertension, diabetes, cardiovascular disease, and depression.",
        "how often should viral load be tested": "WHO recommends viral load testing at 6 months, 12 months, and annually thereafter if suppressed.",
    }

    user_q = st.text_input("Your question:")
    if user_q:
        q_lower = user_q.lower()
        response = None
        for key in knowledge_base:
            if key in q_lower:
                response = knowledge_base[key]
                break
        if response:
            st.success(response)
        else:
            st.warning("Sorry, I don’t have an exact answer. Please consult WHO HIV guidelines.")
