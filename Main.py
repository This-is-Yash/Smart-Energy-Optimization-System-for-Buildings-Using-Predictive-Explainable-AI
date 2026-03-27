# # # ============================================
# # # SMART ENERGY OPTIMIZATION USING REAL DATASET
# # # ============================================

# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt

# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_absolute_error

# # import shap

# # # ============================================
# # # 1. LOAD DATASET
# # # ============================================

# # # Replace with your file path
# # data = pd.read_csv("global-data-on-sustainable-energy (1).csv")

# # print("Dataset Loaded Successfully")
# # print(data.head())

# # # ============================================
# # # 2. DATA PREPROCESSING
# # # ============================================

# # # Drop unnecessary columns
# # data = data.drop(columns=["Entity", "Year"], errors="ignore")

# # # Handle missing values
# # # data = data.fillna(data.mean())
# # # Fill only numeric columns
# # numeric_cols = data.select_dtypes(include=np.number).columns
# # data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# # # ============================================
# # # 3. DEFINE TARGET VARIABLE
# # # ============================================

# # # Target: Energy Consumption per person
# # target = "Primary energy consumption per capita (kWh/person)"

# # X = data.drop(columns=[target])
# # y = data[target]

# # # ============================================
# # # 4. TRAIN-TEST SPLIT
# # # ============================================

# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.2, random_state=42
# # )

# # # ============================================
# # # 5. PREDICTIVE AI MODEL
# # # ============================================

# # model = RandomForestRegressor(n_estimators=200)
# # model.fit(X_train, y_train)

# # preds = model.predict(X_test)

# # print("\nModel Performance:")
# # print("MAE:", mean_absolute_error(y_test, preds))

# # # ============================================
# # # 6. EXPLAINABLE AI (SHAP)
# # # ============================================

# # explainer = shap.TreeExplainer(model)
# # shap_values = explainer.shap_values(X_test)

# # # Feature Importance Plot
# # shap.summary_plot(shap_values, X_test)

# # # ============================================
# # # 7. OPTIMIZATION ENGINE (SMART LOGIC)
# # # ============================================

# # def optimize_energy(row):
# #     suggestions = []

# #     if row["Electricity from fossil fuels (TWh)"] > 1000:
# #         suggestions.append("Reduce fossil fuel usage, shift to renewables")

# #     if row["Renewable energy share in total final energy consumption (%)"] < 20:
# #         suggestions.append("Increase renewable energy adoption")

# #     if row["Energy intensity level of primary energy (MJ/$2011 PPP GDP)"] > 5:
# #         suggestions.append("Improve energy efficiency in industries")

# #     if row["Value_co2_emissions (metric tons per capita)"] > 5:
# #         suggestions.append("Implement carbon reduction strategies")

# #     return suggestions

# # # Apply optimization
# # X_test["Recommendations"] = X_test.apply(optimize_energy, axis=1)

# # # ============================================
# # # 8. OUTPUT SAMPLE
# # # ============================================

# # print("\nSample Results:\n")

# # for i in range(5):
# #     print("Input:", X_test.iloc[i].to_dict())
# #     print("Predicted Energy:", preds[i])
# #     print("Suggestions:", X_test.iloc[i]["Recommendations"])
# #     print("------")
# # ============================================
# # SMART ENERGY OPTIMIZATION SYSTEM (ADVANCED)
# # ============================================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score

# import shap
# from lime.lime_tabular import LimeTabularExplainer

# # ============================================
# # 1. LOAD DATA
# # ============================================

# data = pd.read_csv("global-data-on-sustainable-energy (1).csv")

# print("Dataset Loaded Successfully\n")
# print(data.head())

# # ============================================
# # 2. PREPROCESSING (FIXED ERROR)
# # ============================================

# # Drop non-useful column
# data = data.drop(columns=["Entity"], errors="ignore")

# # Fill missing values only for numeric columns
# numeric_cols = data.select_dtypes(include=np.number).columns
# data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# print("\nData Types:\n", data.dtypes)

# # ============================================
# # 3. TARGET VARIABLE
# # ============================================

# target = "Primary energy consumption per capita (kWh/person)"

# X = data.drop(columns=[target])
# y = data[target]

# # ============================================
# # 4. TRAIN TEST SPLIT
# # ============================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ============================================
# # 5. MODEL TRAINING
# # ============================================

# model = RandomForestRegressor(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)

# preds = model.predict(X_test)

# print("\nModel Performance:")
# print("MAE:", mean_absolute_error(y_test, preds))
# print("R2 Score:", r2_score(y_test, preds))

# # ============================================
# # 6. VISUALIZATION
# # ============================================

# # 🔹 Correlation Heatmap
# plt.figure(figsize=(10,8))
# sns.heatmap(data.corr(), cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.show()

# # 🔹 Actual vs Predicted
# plt.figure()
# plt.scatter(y_test, preds)
# plt.xlabel("Actual Energy")
# plt.ylabel("Predicted Energy")
# plt.title("Actual vs Predicted")
# plt.show()

# # 🔹 Feature Importance
# importances = model.feature_importances_
# features = X.columns

# plt.figure()
# plt.barh(features, importances)
# plt.title("Feature Importance")
# plt.show()

# # ============================================
# # 7. EXPLAINABLE AI - SHAP
# # ============================================

# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)

# print("\nGenerating SHAP plots...")

# shap.summary_plot(shap_values, X_test)
# shap.plots.bar(shap.Explanation(values=shap_values, data=X_test.values, feature_names=X_test.columns))

# # ============================================
# # 8. EXPLAINABLE AI - LIME
# # ============================================

# print("\nGenerating LIME explanation...")

# lime_explainer = LimeTabularExplainer(
#     X_train.values,
#     feature_names=X.columns.tolist(),
#     mode="regression"
# )

# # Explain one instance
# i = 0
# lime_exp = lime_explainer.explain_instance(
#     X_test.iloc[i].values,
#     model.predict
# )

# lime_exp.show_in_notebook(show_table=True)

# # ============================================
# # 9. SIMPLE PDP (PARTIAL DEPENDENCE STYLE)
# # ============================================

# feature = X.columns[0]

# plt.figure()
# plt.scatter(X_test[feature], preds)
# plt.xlabel(feature)
# plt.ylabel("Predicted Energy")
# plt.title(f"Effect of {feature} on Energy")
# plt.show()

# # ============================================
# # 10. OPTIMIZATION ENGINE
# # ============================================

# def optimize_energy(row):
#     suggestions = []

#     if row["Electricity from fossil fuels (TWh)"] > 1000:
#         suggestions.append("Reduce fossil fuel usage")

#     if row["Renewable energy share in total final energy consumption (%)"] < 20:
#         suggestions.append("Increase renewable energy")

#     if row["Energy intensity level of primary energy (MJ/$2011 PPP GDP)"] > 5:
#         suggestions.append("Improve efficiency")

#     if row["Value_co2_emissions (metric tons per capita)"] > 5:
#         suggestions.append("Reduce CO2 emissions")

#     return suggestions

# X_test["Recommendations"] = X_test.apply(optimize_energy, axis=1)

# # ============================================
# # 11. OUTPUT SAMPLE
# # ============================================

# print("\nSample Results:\n")

# for i in range(5):
#     print("Predicted Energy:", preds[i])
#     print("Recommendations:", X_test.iloc[i]["Recommendations"])
#     print("------")
# ============================================
# SMART ENERGY OPTIMIZATION SYSTEM (FINAL)
# ============================================

# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import shap
from lime.lime_tabular import LimeTabularExplainer

import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. LOAD DATA
# ============================================

file_path = "global-data-on-sustainable-energy (1).csv"  # change if needed
data = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully\n")
print(data.head())

# ============================================
# 2. DATA CLEANING (IMPORTANT FIXES)
# ============================================

# Remove unwanted column
data = data.drop(columns=["Entity"], errors="ignore")

# Fix column names (remove newline + spaces)
data.columns = data.columns.str.replace("\n", " ").str.strip()

# 🔥 Remove commas from ALL columns (like "8,358")
for col in data.columns:
    data[col] = data[col].astype(str).str.replace(",", "")

# Convert everything to numeric safely
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Fill missing values
data = data.fillna(data.mean())

print("\n✅ Cleaned Data Types:\n")
print(data.dtypes)

# Check if any null remains
print("\nRemaining Null Values:\n", data.isnull().sum())

# ============================================
# 3. DEFINE TARGET
# ============================================

target = "Primary energy consumption per capita (kWh/person)"

if target not in data.columns:
    raise Exception(f"❌ Target column not found. Available columns:\n{data.columns}")

X = data.drop(columns=[target])
y = data[target]

# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 5. MODEL TRAINING
# ============================================

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("MAE:", mean_absolute_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))

# ============================================
# 6. VISUALIZATIONS
# ============================================

# 🔹 Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 🔹 Actual vs Predicted
plt.figure()
plt.scatter(y_test, preds)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted")
plt.show()

# 🔹 Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()

# ============================================
# 7. EXPLAINABLE AI - SHAP
# ============================================

print("\n🔍 Running SHAP...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# ============================================
# 8. EXPLAINABLE AI - LIME
# ============================================

print("\n🔍 Running LIME...")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    mode="regression"
)

# Explain first test sample
i = 0
lime_exp = lime_explainer.explain_instance(
    X_test.iloc[i].values,
    model.predict
)

print("\nLIME Explanation:")
print(lime_exp.as_list())

# ============================================
# 9. SIMPLE PDP (FEATURE EFFECT)
# ============================================

feature = X.columns[0]

plt.figure()
plt.scatter(X_test[feature], preds)
plt.xlabel(feature)
plt.ylabel("Predicted Energy")
plt.title(f"Effect of {feature} on Prediction")
plt.show()

# ============================================
# 10. OPTIMIZATION ENGINE
# ============================================

def optimize_energy(row):
    suggestions = []

    if row.get("Electricity from fossil fuels (TWh)", 0) > 1000:
        suggestions.append("Reduce fossil fuel usage")

    if row.get("Renewable energy share in total final energy consumption (%)", 0) < 20:
        suggestions.append("Increase renewable energy usage")

    if row.get("Energy intensity level of primary energy (MJ/$2011 PPP GDP)", 0) > 5:
        suggestions.append("Improve energy efficiency")

    if row.get("Value_co2_emissions (metric tons per capita)", 0) > 5:
        suggestions.append("Reduce CO2 emissions")

    return suggestions

X_test["Recommendations"] = X_test.apply(optimize_energy, axis=1)

# ============================================
# 11. FINAL OUTPUT
# ============================================

print("\n📌 SAMPLE RESULTS:\n")

for i in range(5):
    print(f"Prediction {i+1}:")
    print("Predicted Energy:", preds[i])
    print("Recommendations:", X_test.iloc[i]["Recommendations"])
    print("------")