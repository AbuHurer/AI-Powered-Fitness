# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fitness Program Recommender", layout="wide")
st.title("🏋️ Personalized Workout Program Recommender")

# ==========================
# Load assets
# ==========================
try:
    st.write("📂 Loading data and models...")
    df = pd.read_csv("fitness_and_workout_dataset_cleaned.csv")  # CHANGE if needed
    label_encoders = joblib.load("label_encoders.pkl")
    reg = joblib.load("time_per_workout_regressor.pkl")
    st.success("Data and models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# ==========================
# SHAP setup
# ==========================
@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(reg)

# ==========================
# UI: User Inputs
# ==========================
st.sidebar.header("Enter Your Preferences")
user_input = {}

for col in df.columns:
    if col == "time_per_workout":
        continue
    if df[col].dtype == "object":
        # Limit user selection to trained categories
        user_input[col] = st.sidebar.selectbox(col, label_encoders[col].classes_)
    else:
        min_val, max_val = int(df[col].min()), int(df[col].max())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, min_val)

# ==========================
# Prediction & Recommendations
# ==========================
if st.sidebar.button("Get Recommendations"):
    input_df = pd.DataFrame([user_input])

    # --------------------------
    # Encode categorical inputs safely
    # --------------------------
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'Other')
            # Ensure 'Other' exists in encoder classes
            if 'Other' not in le.classes_:
                le_classes = list(le.classes_) + ['Other']
                le.classes_ = np.array(le_classes)
            input_df[col] = le.transform(input_df[col])

    # --------------------------
    # Ensure all features expected by the model exist
    # --------------------------
    feature_cols = reg.feature_names_in_
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = df[col].median()  # fill with median from training data

    # --------------------------
    # Predict time per workout
    # --------------------------
    predicted_time = reg.predict(input_df)[0]
    st.subheader("🔮 Prediction")
    st.write(f"**Predicted Time per Workout:** {predicted_time:.2f} mins")

    # --------------------------
    # Show closest matching programs
    # --------------------------
    st.subheader("💡 Recommended Programs")
    closest_matches = df.iloc[(df["time_per_workout"] - predicted_time).abs().argsort()[:5]]
    st.dataframe(closest_matches)

    # --------------------------
    # SHAP Explanation
    # --------------------------
    st.subheader("📊 Why this was predicted (SHAP)")
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)
