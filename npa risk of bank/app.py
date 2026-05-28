"""
==========================================================
 AI NPA System — Production Multi-Page Web Application
==========================================================
 Advanced Flask backend powering a complete PowerBI-style 
 AI dashboard for Non-Performing Asset prediction.
==========================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ----------------------------------------------------------
# 1. Initialize Application
# ----------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'npa-premium-dashboard-key'

# ----------------------------------------------------------
# 2. Load ML Assets & Dataset
# ----------------------------------------------------------
try:
    with open("model/npa_model.pkl", "rb") as f:
        model = pickle.load(f)
    print(" [OK] Random Forest Model loaded.")

    with open("model/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    print(" [OK] Encoders loaded.")

    # Load broader dataset for advanced PowerBI analytics
    cols_to_use = ["Loan_Amount", "Loan_Type", "Credit_Score", "Repayment_History", "Default_Status"]
    df = pd.read_csv("dataset/npa_dataset.csv", usecols=cols_to_use)
    print(f" [OK] Dataset loaded for analytics ({len(df):,} records).")
except Exception as e:
    print(f" [ERROR] ERROR loading ML assets: {e}")

# ----------------------------------------------------------
# 3. Helper Functions
# ----------------------------------------------------------
def generate_risk_explanations(inputs):
    """Analyzes input values to generate dynamic risk factors."""
    explanations = []
    def add_exp(icon, text, typ): explanations.append({"icon": icon, "text": text, "type": typ})

    score = inputs['credit_score']
    if score < 600: add_exp("fa-solid fa-triangle-exclamation", f"Critical: Low credit score ({score}) indicates poor financial health", "danger")
    elif score < 700: add_exp("fa-solid fa-bell", f"Warning: Average credit score ({score}) requires close monitoring", "warning")
    else: add_exp("fa-solid fa-check-circle", f"Safe: Excellent credit score ({score}) strongly mitigates risk", "safe")

    amount = inputs['loan_amount']
    if amount > 500000: add_exp("fa-solid fa-money-bill-trend-up", f"Factor: High loan principal (₹{amount:,.0f}) inherently carries more default risk", "warning")
    else: add_exp("fa-solid fa-piggy-bank", f"Safe: Moderate loan principal (₹{amount:,.0f}) is manageable", "safe")

    repayment = inputs['repayment_history']
    if repayment < 50: add_exp("fa-solid fa-calendar-xmark", f"Critical: Historical repayment rate is very low ({repayment}%)", "danger")
    elif repayment < 75: add_exp("fa-solid fa-calendar-minus", f"Warning: Historical repayment rate is suboptimal ({repayment}%)", "warning")
    else: add_exp("fa-solid fa-calendar-check", f"Safe: Strong historical repayment compliance ({repayment}%)", "safe")

    return explanations

# ----------------------------------------------------------
# 4. Page Routing
# ----------------------------------------------------------
@app.route("/")
def home():
    loan_types = list(encoders["Loan_Type"].classes_)
    return render_template("index.html", loan_types=loan_types, active_page='home')

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", active_page='dashboard')

@app.route("/about")
def about():
    return render_template("about.html", active_page='about')

@app.route("/model")
def model_info():
    return render_template("model.html", active_page='model')

@app.route("/developer")
def developer():
    return render_template("developer.html", active_page='developer')

# ----------------------------------------------------------
# 5. Prediction Engine
# ----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = {
            "loan_amount": float(request.form["loan_amount"]),
            "loan_type": request.form["loan_type"],
            "credit_score": int(request.form["credit_score"]),
            "repayment_history": float(request.form["repayment_history"]),
            "collateral_value": float(request.form["collateral_value"]),
            "loan_tenure": int(request.form["loan_tenure"])
        }

        loan_type_encoded = encoders["Loan_Type"].transform([inputs["loan_type"]])[0]
        features = np.array([[
            inputs["loan_amount"], loan_type_encoded, inputs["credit_score"],
            inputs["repayment_history"], inputs["collateral_value"], inputs["loan_tenure"]
        ]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        default_prob = round(probability[1] * 100, 1)

        if default_prob >= 75.0:
            risk = {"level": "CRITICAL RISK", "badge": "DANGER", "class": "danger", "color": "#ef4444", "icon": "fa-solid fa-skull-crossbones"}
        elif prediction == 1 or default_prob > 50.0:
            risk = {"level": "HIGH RISK", "badge": "WARNING", "class": "warning", "color": "#f59e0b", "icon": "fa-solid fa-triangle-exclamation"}
        else:
            risk = {"level": "LOW RISK", "badge": "SAFE", "class": "safe", "color": "#10b981", "icon": "fa-solid fa-shield-check"}

        explanations = generate_risk_explanations(inputs)
        return render_template("result.html", risk=risk, default_prob=default_prob, explanations=explanations, inputs=inputs, active_page='home')
    except Exception as e:
        return render_template("result.html", risk={"level": "SYSTEM ERROR", "class": "danger", "icon": "fa-solid fa-circle-xmark"}, default_prob=0, explanations=[], inputs={}, active_page='home')

# ----------------------------------------------------------
# 6. Advanced Analytics APIs (PowerBI Integration)
# ----------------------------------------------------------
@app.route("/api/dashboard_data", methods=["GET"])
def dashboard_data():
    """Returns static global dashboard metrics for the full-width PowerBI layout."""
    try:
        if len(df) == 0:
            return jsonify({"error": "Global dataset is empty."}), 404

        # 1. KPIs
        total_loans = len(df)
        total_defaults = int(df["Default_Status"].sum())
        npa_ratio = round((total_defaults / total_loans) * 100, 1)
        avg_credit = int(df["Credit_Score"].mean())
        avg_loan = int(df["Loan_Amount"].mean())

        kpis = {
            "total_loans": f"{total_loans:,}",
            "total_defaults": f"{total_defaults:,}",
            "npa_ratio": float(npa_ratio),
            "avg_credit": avg_credit,
            "avg_loan": f"{avg_loan:,.0f}"
        }
        
        # 2. Charts Data
        # A. Loan Type vs Default (Bar)
        sector_group = df.groupby("Loan_Type")["Default_Status"].agg(["count", "sum"])
        sector_group["Safe"] = sector_group["count"] - sector_group["sum"]
        chart_loan_type = {
            "labels": list(sector_group.index),
            "default": list(sector_group["sum"].astype(int)),
            "safe": list(sector_group["Safe"].astype(int))
        }

        # B. Asset Viability (Donut)
        chart_donut = {
            "labels": ["Performing Assset", "Non-Performing Asset (NPA)"],
            "values": [total_loans - total_defaults, total_defaults]
        }

        # C. Feature Importance
        # Extract features directly from the loaded ML model
        importances = model.feature_importances_
        feature_names = ["Loan_Amount", "Loan_Type", "Credit_Score", "Repayment_History", "Collateral_Value", "Loan_Tenure"]
        sorted_idx = np.argsort(importances)[::-1]
        chart_importance = {
            "labels": [feature_names[i] for i in sorted_idx],
            "values": [round(float(importances[i]*100), 2) for i in sorted_idx]
        }

        # D. Credit Score Distribution
        bins = [300, 450, 600, 750, 900]
        labels = ["300-450", "451-600", "601-750", "751-900"]
        df["Credit_Bin"] = pd.cut(df["Credit_Score"], bins=bins, labels=labels, include_lowest=True)
        credit_dist = df.groupby(["Credit_Bin", "Default_Status"]).size().unstack(fill_value=0)
        chart_credit_dist = {
            "labels": labels,
            "safe": list(credit_dist.get(0, pd.Series([0]*4)).astype(int)),
            "default": list(credit_dist.get(1, pd.Series([0]*4)).astype(int))
        }

        charts = {
            "loan_type": chart_loan_type,
            "donut": chart_donut,
            "importance": chart_importance,
            "credit_dist": chart_credit_dist
        }

        # 3. Dynamic Intelligent Insights
        insights = []
        insights.append({"type": "danger", "icon": "fa-solid fa-triangle-exclamation", "text": "Low credit score (<600) radically increases default probability across all sectors."})
        insights.append({"type": "warning", "icon": "fa-solid fa-clock-rotate-left", "text": "Poor historical repayment reliability acts as the strongest secondary predictor of systemic defaults."})
        insights.append({"type": "warning", "icon": "fa-solid fa-magnifying-glass-chart", "text": "Micro-principal exposures (< ₹5,00,000) show disproportionately high volatility during economic downturns."})
        
        # Determine highest risk sector safely
        if not sector_group.empty and 'count' in sector_group.columns and 'sum' in sector_group.columns:
             safe_sg = sector_group[sector_group['count'] > 0]
             if not safe_sg.empty:
                  risk_rates = (safe_sg["sum"] / safe_sg["count"]) * 100
                  highest_risk_name = risk_rates.idxmax()
                  highest_rate = round(risk_rates.max(), 1)
                  insights.append({"type": "danger", "icon": "fa-solid fa-building-columns", "text": f"{highest_risk_name} portfolios present the sharpest systemic risk ratio at exactly {highest_rate}%."})
        
        # 4. Detailed Summary Table
        table_group = df.groupby("Loan_Type").agg(
            Count=("Default_Status", "count"),
            Defaults=("Default_Status", "sum"),
            AvgCredit=("Credit_Score", "mean"),
            AvgLoan=("Loan_Amount", "mean")
        ).round(1).reset_index()
        
        table_rows = []
        for _, row in table_group.iterrows():
            table_rows.append({
                "type": row["Loan_Type"],
                "total": int(row["Count"]),
                "defaults": int(row["Defaults"]),
                "avg_credit": int(row["AvgCredit"]),
                "avg_loan": f"₹{row['AvgLoan']:,.0f}"
            })

        return jsonify({"kpis": kpis, "charts": charts, "insights": insights, "table": table_rows})
    except Exception as e:
        print(f"[API ERROR]: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
