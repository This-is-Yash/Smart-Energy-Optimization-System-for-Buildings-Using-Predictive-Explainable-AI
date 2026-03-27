
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


file_path = "global-data-on-sustainable-energy (1).csv"  # change if needed
data = pd.read_csv(file_path)

print("Dataset Loaded Successfully\n")
print(data.head())


data = data.drop(columns=["Entity"], errors="ignore")

data.columns = data.columns.str.replace("\n", " ").str.strip()

for col in data.columns:
    data[col] = data[col].astype(str).str.replace(",", "")

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.fillna(data.mean())

print("\nCleaned Data Types:\n")
print(data.dtypes)

print("\nRemaining Null Values:\n", data.isnull().sum())

target = "Primary energy consumption per capita (kWh/person)"

if target not in data.columns:
    raise Exception(f" Target column not found. Available columns:\n{data.columns}")

X = data.drop(columns=[target])
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\n MODEL PERFORMANCE")
print("MAE:", mean_absolute_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure()
plt.scatter(y_test, preds)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted")
plt.show()

importances = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()


print("\nRunning SHAP...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")


print("\n Running LIME...")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    mode="regression"
)

i = 0
lime_exp = lime_explainer.explain_instance(
    X_test.iloc[i].values,
    model.predict
)

print("\nLIME Explanation:")
print(lime_exp.as_list())


feature = X.columns[0]

plt.figure()
plt.scatter(X_test[feature], preds)
plt.xlabel(feature)
plt.ylabel("Predicted Energy")
plt.title(f"Effect of {feature} on Prediction")
plt.show()


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


print("\nSAMPLE RESULTS:\n")

for i in range(5):
    print(f"Prediction {i+1}:")
    print("Predicted Energy:", preds[i])
    print("Recommendations:", X_test.iloc[i]["Recommendations"])
    print("------")
