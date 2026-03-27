import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import shap
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="⚡ AI Dashboard", layout="wide")
st.title("Smart Energy Optimization System for Buildings Using Predictive & Explainable AI")

# ==============================
# 🔥 SESSION STATE
# ==============================
for key in ["data", "model", "trained", "X_train", "y_train", "X_test", "y_test", "task"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================
# 📂 FILE UPLOAD
# ==============================
files = st.file_uploader("📂 Upload CSV files", type=["csv"], accept_multiple_files=True)

if files:
    st.session_state.data = pd.concat(
        [pd.read_csv(f) for f in files], ignore_index=True
    )

data = st.session_state.data

# ==============================
# 📊 DATA PREVIEW
# ==============================
if data is not None:
    st.subheader("📊 Data Preview")
    st.dataframe(data.head())

    target = st.selectbox("🎯 Select Target Column", data.columns)

    data = data.dropna(subset=[target])

    X = data.drop(columns=[target])
    y = data[target]

    X = X.fillna(X.mean(numeric_only=True))

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object" or y.nunique() < 10:
        y = LabelEncoder().fit_transform(y.astype(str))
        task = "classification"
    else:
        task = "regression"

    st.session_state.task = task

    # ==============================
    # 🚀 TRAIN MODEL
    # ==============================
    if st.button("🚀 Train Model") or st.session_state.trained:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.trained = True
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.success("✅ Model Trained Successfully!")

# ==============================
# 🔮 PREDICTION + KPI + VISUALS
# ==============================
if st.session_state.trained:

    st.subheader("🔮 Live Prediction")

    input_data = {}

    for col in st.session_state.X_train.columns:
        val = st.slider(
            col,
            float(st.session_state.X_train[col].min()),
            float(st.session_state.X_train[col].max()),
            float(st.session_state.X_train[col].mean())
        )
        input_data[col] = val

    input_df = pd.DataFrame([input_data])
    prediction = st.session_state.model.predict(input_df)[0]

    st.success(f"🎯 Prediction: {prediction}")

    # ==============================
    # 📊 KPI METRICS
    # ==============================
    st.subheader("📊 Model Performance")

    y_pred = st.session_state.model.predict(st.session_state.X_test)

    col1, col2 = st.columns(2)

    if st.session_state.task == "regression":
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        r2 = r2_score(st.session_state.y_test, y_pred)

        col1.metric("MSE", round(mse, 2))
        col2.metric("R² Score", round(r2, 2))

    else:
        acc = accuracy_score(st.session_state.y_test, y_pred)
        col1.metric("Accuracy", round(acc, 2))

    # ==============================
    # 📈 ACTUAL VS PREDICTED
    # ==============================
    st.subheader("📈 Actual vs Predicted")

    fig = px.scatter(
        x=st.session_state.y_test,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"}
    )
    st.plotly_chart(fig)

    # ==============================
    # 📉 RESIDUALS (REGRESSION ONLY)
    # ==============================
    if st.session_state.task == "regression":
        st.subheader("📉 Residual Distribution")

        residuals = st.session_state.y_test - y_pred
        fig = px.histogram(residuals)
        st.plotly_chart(fig)

    # ==============================
    # 📊 SHAP
    # ==============================
    st.subheader("📊 SHAP Feature Importance")

    try:
        explainer = shap.TreeExplainer(st.session_state.model)
        shap_values = explainer(st.session_state.X_train, check_additivity=False)

        shap.summary_plot(shap_values, st.session_state.X_train, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    except Exception as e:
        st.warning(f"SHAP failed: {e}")

    # ==============================
    # 💡 LIME (FIXED OUTPUT)
    # ==============================
    st.subheader("💡 LIME Explanation")

    try:
        mode = "classification" if st.session_state.task == "classification" else "regression"

        explainer_lime = LimeTabularExplainer(
            st.session_state.X_train.values,
            feature_names=st.session_state.X_train.columns,
            mode=mode
        )

        lime_exp = explainer_lime.explain_instance(
            input_df.values[0],
            st.session_state.model.predict
        )

        # 🔥 Convert to readable text instead of JSON
        for feature, weight in lime_exp.as_list():
            st.write(f"👉 {feature} → impact: {round(weight,3)}")

    except Exception as e:
        st.warning(f"LIME failed: {e}")

    # ==============================
    # 🧠 FINAL INSIGHT
    # ==============================
    st.subheader("🧠 Final Insight")

    st.info(
        "📌 The model predicts based on patterns learned from historical data. "
        "SHAP shows global importance while LIME explains this specific prediction, "
        "giving a rule-based understanding of feature influence."
    )

# ==============================
# 📊 CUSTOM VISUALIZATION
# ==============================
if data is not None:
    st.subheader("📊 Custom Visualization")

    col1 = st.selectbox("X-axis", data.columns)
    col2 = st.selectbox("Y-axis", data.columns)

    fig = px.scatter(data, x=col1, y=col2)
    st.plotly_chart(fig)