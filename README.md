# Healthcare_prediction
# 🏥 XGBoost Medical Diagnostics Predictor

An attractive, production-ready web application built using **Flask** and **Tailwind CSS** that deploys a machine learning model to predict healthcare risk classes based on patient diagnostic parameters. 

🌐 **Live Application URL:** [Healthcare Prediction Portal](https://healthcare-prediction-1-9230.onrender.com/)

---

## 🚀 Project Overview

This system wraps an optimized `XGBoost Classifier` model into an intuitive web interface. While machine learning architectures handle numerical data behind the scenes, this application bridges the gap for end-users by handling raw text and categorical parameters in a modern dashboard interface and automatically translating them into data the engine understands.

### 📊 Feature Pipeline Architecture
The model evaluates predictions based on **9 essential healthcare features** in a specific configuration sequence:
1. **Age** (Numerical)
2. **Gender** (Converted from Categorical strings to Numerical mappings)
3. **Blood Type** (Converted from Categorical strings to Numerical mappings)
4. **Medical Condition** (Converted from Categorical strings to Numerical mappings)
5. **Hospital Site** (Converted from Categorical strings to Numerical mappings)
6. **Insurance Provider** (Converted from Categorical strings to Numerical mappings)
7. **Billing Amount** (Numerical)
8. **Admission Type** (Converted from Categorical strings to Numerical mappings)
9. **Medication** (Converted from Categorical strings to Numerical mappings)

---

## 🛠️ File Layout

```text
.
├── app.py                 # Self-contained Flask backend & Tailwind HTML dashboard
├── xgboost.pkl            # Trained Serialization file for XGBoost Classifier 
└── requirements.txt       # Production dependencies for server environment
