# ⚡ AI-Assisted Transmission Line Design System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn\&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-Regression-008000?logo=xgboost\&logoColor=white)]()


## 📘 Overview

This is a **machine learning-powered transmission line design system** built with **Python** and **Streamlit**.
It helps electrical engineers choose the right **conductors**, **insulators**, and estimate **project cost** and **power losses** for high-voltage transmission lines.

The system uses two ML models:

1. **RandomForestClassifier**
   Predicts conductor type (AAAC / ACSR / ACCC) and insulator type (Porcelain / Glass / Composite).

2. **XGBoost Regressor**
   Estimates total project cost and line losses based on line length, number of towers, conductor type, and regional factors.

The Streamlit app includes multiple interactive pages for data input, predictions, visuals, and model performance.


## 🏗️ System Architecture

### 🎨 Frontend Architecture

* **Framework:** Streamlit
* **Layout:** Wide layout with sidebar navigation
* **Pages:**

  * Dashboard / Overview
  * Conductor & Insulator Selection
  * Cost & Loss Estimation
  * Results & Reports
  * Model Performance

**Why Streamlit?**
It allows fast development and deployment without needing frontend frameworks. Keeping navigation inside one file makes state management simple.


### 🧠 Backend Architecture

* **Training Pipeline:** `train_models.py` handles offline training
* **Model Serving:** Cached loading with `@st.cache_resource`
* **State Management:** `st.session_state` keeps values between pages

**Why separate training?**
It lets you retrain models anytime without changing the app itself. Caching avoids reloading models on every user action.

---

## 🤖 Machine Learning Models

### 🔌 Conductor & Insulator Selection Model

* **Algorithm:** RandomForestClassifier (scikit-learn)
* **Inputs:**

  * Voltage (66–800 kV)
  * Current (100–5000 A)
  * Temperature (−10 to 60°C)
  * Pollution level (Low / Medium / High)
* **Outputs:** Conductor type, Insulator type
* **Preprocessing:** OneHotEncoder + ColumnTransformer
* **Training Data:** 100,000 synthetic samples (rule-based labels)

**Why RandomForest?**
It's easy to understand, works well with mixed features, and performs reliably on structured engineering data.

---

### 💰 Cost & Loss Estimation Model

* **Algorithm:** XGBRegressor (wrapped in MultiOutputRegressor)
* **Inputs:**

  * Route length (10–1000 km)
  * Tower count (10–500)
  * Conductor type
  * Region factor (0.8–2.0)
* **Outputs:**

  * Total project cost
  * Line loss percentage
* **Preprocessing:** OneHotEncoder
* **Training Data:** 100,000 synthetic samples

**Why XGBoost?**
It performs strongly on tabular regression tasks and handles complex patterns better than traditional ML algorithms.

---

## 📦 Data & Storage

### 🗂️ Data Handling

* **Models:** Saved as `.joblib` files
* **App State:** Managed in memory with Streamlit
* **No Database:** The app does not store user history or accounts

**Why no database?**
This tool only calculates results—it doesn’t need long-term storage.

---

## 📏 Engineering Calculations

### 🌫️ Creepage Distance

```text
creepage = voltage × pollution_factor
pollution_factors = {
    'Low': 20,
    'Medium': 25,
    'High': 31
}
```

A domain-accurate formula ensures realistic insulator sizing.

---

## 📚 External Dependencies

### 🐍 Python Libraries

* **streamlit** – UI framework
* **pandas** – Data handling
* **numpy** – Numerical operations
* **plotly.express** – Visual charts
* **scikit-learn** – RandomForest + preprocessing
* **xgboost** – Regression model
* **joblib** – Model saving/loading

### 🖼️ Assets

* `logo.png` – Branding image (90px width)

### 🧪 Generated Model Files

* `conductor_model.joblib` – RandomForest pipeline
* `cost_model.joblib` – XGBoost pipeline

**Important:**
You must run `train_models.py` at least once to generate these model files before launching the Streamlit app.
