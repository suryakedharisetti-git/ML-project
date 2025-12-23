import streamlit as st
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path
import os

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

mongo_uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"

# ---------------------------
# Detect Streamlit Cloud
# ---------------------------
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_RUNTIME_ENV") is not None

# ---------------------------
# MongoDB (LOCAL ONLY)
# ---------------------------
collection = None

if not IS_STREAMLIT_CLOUD:
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]
    except Exception:
        collection = None  # silently ignore MongoDB errors

# ---------------------------
# Load dataset safely
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hidden_hunger.csv"

# Optional fallback
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR.parent / "data" / "hidden_hunger.csv"

df = pd.read_csv(DATA_PATH)

# ---------------------------
# UI
# ---------------------------
st.title("🌾 NutriScope — Hidden Hunger Detection App")
st.markdown("### Enter the following nutritional and demographic details:")

# Average nutrient values
avg_vitamin_a = df["Vitamin_A_Intake_ug"].mean()
avg_vitamin_d = df["Vitamin_D_Intake_IU"].mean()
avg_zinc = df["Zinc_Intake_mg"].mean()
avg_iron = df["Iron_Intake_mg"].mean()
avg_folate = df["Folate_Intake_ug"].mean()

# ---------------------------
# User Inputs
# ---------------------------
age = st.number_input("Age", min_value=1, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.selectbox("Income Bracket", ["low", "lower_middle", "upper_middle", "high"])
education = st.selectbox("Education Level", ["primary", "secondary", "tertiary"])

vitamin_a = st.number_input("Vitamin A Intake (µg)", min_value=0.0, step=1.0)
vitamin_d = st.number_input("Vitamin D Intake (IU)", min_value=0.0, step=1.0)
zinc = st.number_input("Zinc Intake (mg)", min_value=0.0, step=0.1)
iron = st.number_input("Iron Intake (mg)", min_value=0.0, step=0.1)
folate = st.number_input("Folate Intake (µg)", min_value=0.0, step=1.0)

# ---------------------------
# Prediction & Analysis
# ---------------------------
if st.button("🔍 Predict Hidden Hunger Risk"):

    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Income_Bracket": income,
        "Education_Level": education,
        "Vitamin_A_Intake_ug": vitamin_a,
        "Vitamin_D_Intake_IU": vitamin_d,
        "Zinc_Intake_mg": zinc,
        "Iron_Intake_mg": iron,
        "Folate_Intake_ug": folate,
    }])

    st.subheader("📋 Input Data Preview")
    st.dataframe(input_df)

    # Ratios
    ratios = {
        "Vitamin A": vitamin_a / avg_vitamin_a,
        "Vitamin D": vitamin_d / avg_vitamin_d,
        "Zinc": zinc / avg_zinc,
        "Iron": iron / avg_iron,
        "Folate": folate / avg_folate,
    }

    avg_ratio = sum(ratios.values()) / len(ratios)
    risk_score = max(0, min(1, 1 - avg_ratio))

    st.markdown("### 📊 Risk Analysis")
    st.progress(risk_score)

    if avg_ratio < 0.7:
        st.error(f"⚠ High Risk of Hidden Hunger! (Risk Score: {risk_score:.2f})")
    elif avg_ratio < 0.9:
        st.warning(f"⚠ Moderate Risk of Hidden Hunger. (Risk Score: {risk_score:.2f})")
    else:
        st.success(f"✅ Low Risk of Hidden Hunger. (Risk Score: {risk_score:.2f})")

    st.markdown("### 💡 Detailed Nutrient Feedback:")
    for nutrient, ratio in ratios.items():
        if ratio < 0.7:
            st.write(f"❌ {nutrient}: Very low intake ({ratio*100:.1f}% of avg)")
        elif ratio < 0.9:
            st.write(f"⚠ {nutrient}: Slightly low intake ({ratio*100:.1f}% of avg)")
        else:
            st.write(f"✅ {nutrient}: Healthy intake ({ratio*100:.1f}% of avg)")

    st.info(
        "Tip: Maintain balanced intake of Vitamin A, D, Zinc, Iron, and Folate for optimal nutrition."
    )

    # ---------------------------
    # Save to MongoDB (LOCAL ONLY, SILENT)
    # ---------------------------
    if collection is not None:
        try:
            record = input_df.to_dict("records")[0]
            record["Risk_Score"] = float(risk_score)
            collection.insert_one(record)
        except Exception:
            pass  # never show DB errors to users
