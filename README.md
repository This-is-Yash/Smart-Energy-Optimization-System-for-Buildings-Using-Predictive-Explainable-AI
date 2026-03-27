# Smart Energy Optimization System for Buildings

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red)](https://blh4aiynjiyr5u9drsyejz.streamlit.app/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

## Live Application

Access the deployed application here:  
https://blh4aiynjiyr5u9drsyejz.streamlit.app/

---

## Overview

The Smart Energy Optimization System is a data-driven application designed to analyze global energy consumption patterns and predict renewable energy trends. It integrates machine learning, time series forecasting, and explainable AI techniques into an interactive web interface.

The system allows users to upload datasets, train predictive models, visualize trends, and interpret results with transparency.

---

## Features

### Predictive Modeling
- Train a machine learning model to estimate renewable energy share
- Interactive input interface for real-time predictions
- Based on Random Forest regression

### Performance Evaluation
- Displays Mean Squared Error (MSE)
- Displays R² Score
- Compares actual vs predicted values visually

### Time Series Analysis
- Visualizes historical energy trends
- Applies rolling average smoothing
- Forecasts future values using ARIMA

### Energy Insights
- Identifies peak energy usage
- Computes average consumption
- Helps detect patterns and irregularities

### Explainable AI
- SHAP for global feature importance
- LIME for local prediction interpretation
- Enhances model transparency and trust

---

## Technology Stack

- Frontend: Streamlit  
- Data Processing: Pandas, NumPy  
- Machine Learning: Scikit-learn  
- Time Series Forecasting: Statsmodels (ARIMA)  
- Visualization: Plotly, Matplotlib, Seaborn  
- Explainability: SHAP, LIME  

---

## Dataset

The application uses a global sustainable energy dataset containing:

- Renewable energy metrics  
- Electricity generation data  
- CO₂ emissions  
- GDP and population statistics  
- Link: https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy/data
---
## Screenshot
![shap_value](shap_value.png)
![shape_yearwise](shap_yearwise.png)
![Mean_value](mean_shap.png)
![Figure1](Figure_1.png)
![Figure2](Figure_2.png)
![Figure3](Figure_3.png)

<img width="1898" height="754" alt="Screenshot 2026-03-27 181036" src="https://github.com/user-attachments/assets/4a131e4b-3caa-4269-8a0c-053e6f83069b" />
<img width="1019" height="570" alt="Screenshot 2026-03-27 181024" src="https://github.com/user-attachments/assets/a50fc940-38f1-43fd-a493-8941c7204198" />
<img width="1310" height="411" alt="Screenshot 2026-03-27 181009" src="https://github.com/user-attachments/assets/2c8f218e-c12e-4edd-884d-621b74c921c5" />
<img width="1879" height="779" alt="Screenshot 2026-03-27 181002" src="https://github.com/user-attachments/assets/e8949167-9d91-4c84-a9c0-cc2714d86e90" />
<img width="1856" height="821" alt="Screenshot 2026-03-27 180947" src="https://github.com/user-attachments/assets/5ded2290-8b2f-4a4f-8f3d-1c7971b411a8" />
<img width="1844" height="696" alt="Screenshot 2026-03-27 180603" src="https://github.com/user-attachments/assets/00567e1d-d48b-4554-8578-e658c40d35a5" />
<img width="1917" height="885" alt="Screenshot 2026-03-27 180534" src="https://github.com/user-attachments/assets/f6ab7fff-377b-4f95-a69f-76ee10483e2f" />

---
## Use Cases
- Smart building energy management
- Sustainability and policy analysis
- Academic and research projects
- Data science portfolios

---
## Contributing

Contributions are welcome. Please fork the repository and submit a pull request for review.

---
## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
