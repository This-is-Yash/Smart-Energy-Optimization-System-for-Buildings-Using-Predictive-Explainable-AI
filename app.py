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


st.set_page_config(page_title="Energy AI", layout="wide")
st.title("⚡ Smart Energy Optimization System for Buildings Using Predictive & Explainable AI")

keys = ["data", "model", "trained", "X_train", "y_train", "X_test", "y_test", "task"]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

files = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=True)

if files:
    st.session_state.data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

data = st.session_state.data

if data is not None:

    st.subheader("Data Preview")
    st.dataframe(data.head())

    target = st.selectbox("Select Target Column", data.columns)

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

    if st.button("Train Model") or st.session_state.trained:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.trained = True
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.success("Model Trained!")

if st.session_state.trained:

    model = st.session_state.model
    X_train = st.session_state.X_train

    tabs = st.tabs([
        "Prediction",
        "Performance",
        "Trends",
        "Energy Insights",
        "Explainability"
    ])

    with tabs[0]:

        st.subheader(" Live Prediction")

        input_data = {}

        for col in X_train.columns:
            val = st.slider(
                col,
                float(X_train[col].min()),
                float(X_train[col].max()),
                float(X_train[col].mean())
            )
            input_data[col] = val

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.success(f" Prediction: {prediction}")

    with tabs[1]:

        st.subheader(" Model Performance")

        y_pred = model.predict(st.session_state.X_test)

        col1, col2 = st.columns(2)

        if st.session_state.task == "regression":
            mse = mean_squared_error(st.session_state.y_test, y_pred)
            r2 = r2_score(st.session_state.y_test, y_pred)

            col1.metric("MSE", round(mse, 2))
            col2.metric("R²", round(r2, 2))

        else:
            acc = accuracy_score(st.session_state.y_test, y_pred)
            col1.metric("Accuracy", round(acc, 2))

        fig = px.scatter(
            x=st.session_state.y_test,
            y=y_pred,
            labels={"x": "Actual", "y": "Predicted"}
        )
        st.plotly_chart(fig)

    with tabs[2]:

        st.subheader(" Energy Trends & Forecasting")

        col1, col2 = st.columns(2)

        with col1:
            date_col = st.selectbox(" Select Date Column", ["None"] + list(data.columns))

        with col2:
            value_col = st.selectbox(" Select Energy Column", list(data.columns))

        if date_col != "None":

            try:
                df = data[[date_col, value_col]].copy()
                # df[date_col] = pd.to_datetime(df[date_col], errors='%Y')
                df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y', errors='coerce')
                df = df.dropna().sort_values(by=date_col)

                st.markdown("###  Energy Consumption Trend")

                fig1 = px.line(df, x=date_col, y=value_col)
                st.plotly_chart(fig1, use_container_width=True)

                df["Rolling Avg"] = df[value_col].rolling(window=5).mean()

                st.markdown("###  Smoothed Trend (Rolling Avg)")

                fig2 = px.line(df, x=date_col, y=[value_col, "Rolling Avg"])
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("###  Forecast Future Energy Usage")

                from statsmodels.tsa.arima.model import ARIMA

                forecast_days = st.slider("Forecast Days", 5, 30, 10)

                ts = df.set_index(date_col)[value_col]

                model_arima = ARIMA(ts, order=(2, 1, 2))
                model_fit = model_arima.fit()

                forecast = model_fit.forecast(steps=forecast_days)

                future_dates = pd.date_range(
                    start=ts.index[-1],
                    periods=forecast_days + 1,
                    freq="Y"
                )[1:]

                forecast_df = pd.DataFrame({
                    date_col: future_dates,
                    "Forecast": forecast
                })

                st.markdown("###  Actual vs Forecast")

                fig3 = px.line()

                fig3.add_scatter(
                    x=df[date_col],
                    y=df[value_col],
                    name="Actual"
                )

                fig3.add_scatter(
                    x=forecast_df[date_col],
                    y=forecast_df["Forecast"],
                    name="Forecast"
                )

                st.plotly_chart(fig3, use_container_width=True)

                st.success(" Forecast generated successfully using ARIMA")

            except Exception as e:
                st.error(f" Error: {e}")

        else:
            st.info(" Please select a date column to enable trend analysis")
    with tabs[3]:

        st.subheader("Energy Insights")

        peak = data[target].max()
        avg = data[target].mean()

        st.metric("Peak Usage", round(peak, 2))
        st.metric("Average Usage", round(avg, 2))

        if peak > avg * 1.5:
            st.error("High energy spike detected")
        else:
            st.success("Energy usage stable")
    with tabs[4]:

        st.subheader(" SHAP + LIME")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_train, check_additivity=False)

            shap.summary_plot(shap_values, X_train, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except:
            st.warning("SHAP failed")

        try:
            explainer_lime = LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns,
                mode="classification" if st.session_state.task == "classification" else "regression"
            )

            lime_exp = explainer_lime.explain_instance(
                input_df.values[0],
                model.predict
            )

            for f, w in lime_exp.as_list():
                st.write(f" {f} → impact {round(w,3)}")

        except:
            st.warning("LIME failed")
