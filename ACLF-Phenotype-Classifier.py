import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. Scaling Parameters & Centroids from MIMIC ---
SCALING = {
    "ph_min.y": {"mean": 7.29, "sd": 0.131, "label": "pH (min)"},
    "bicarbonate_min.y": {"mean": 18.3, "sd": 5.63, "label": "Bicarbonate (min)"},
    "lactate_max.y": {"mean": 4.59, "sd": 4.59, "label": "Lactate (max)"},
    "anion_gap_max": {"mean": 19.6, "sd": 7.01, "label": "Anion Gap (max)"},
    "mbp_mean": {"mean": 72.8, "sd": 10.0, "label": "MBP (mean)"},
    "heart_rate_mean": {"mean": 89.3, "sd": 17.0, "label": "Heart Rate (mean)"},
    "resp_rate_mean": {"mean": 19.9, "sd": 4.60, "label": "Resp Rate (mean)"},
    "creatinine_max.y": {"mean": 2.88, "sd": 2.05, "label": "Creatinine (max)"},
    "bun_max.y": {"mean": 49.8, "sd": 32.2, "label": "BUN (max)"},
    "tbil_max": {"mean": 9.22, "sd": 10.8, "label": "Total Bilirubin (max)"},
    "inr_max.y": {"mean": 2.44, "sd": 1.35, "label": "INR (max)"},
    "platelet_min": {"mean": 109.0, "sd": 83.5, "label": "Platelets (min)"},
    "wbc_max.y": {"mean": 15.1, "sd": 11.7, "label": "WBC (max)"},
    "temperature_mean": {"mean": 36.6, "sd": 0.60, "label": "Temperature (mean)"},
    "pao2fio2ratio_min": {"mean": 182.0, "sd": 112.0, "label": "PaO2/FiO2 (min)"}
}

CENTROIDS = {
    "Phenotype 1": [0.450, 0.506, -0.433, -0.527, 0.0868, -0.354, -0.349, -0.252, -0.224, -0.052, -0.165, 0.063, -0.075, 0.045, 0.125],
    "Phenotype 2": [-0.574, -0.645, 0.553, 0.673, -0.111, 0.451, 0.445, 0.321, 0.286, 0.066, 0.210, -0.081, 0.096, -0.057, -0.160]
}

# --- 2. Page Configuration ---
st.set_page_config(page_title="ACLF Phenotype Classifier", layout="wide")
st.markdown("""
    <style>
    .metric-card { padding: 15px; border-radius: 10px; background: #f8f9fa; border: 1px solid #ddd; }
    .stNumberInput label { font-size: 0.9rem; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Sidebar: Clinical Input ---
with st.sidebar:
    st.header("📋 Patient Clinical Profile")
    st.write("Enter values recorded within 24h of admission.")
    
    user_raw = {}
    
    with st.expander("🧪 Acid-Base & Metabolic", expanded=True):
        user_raw["ph_min.y"] = st.number_input("pH (min)", 6.50, 8.00, 7.35)
        user_raw["bicarbonate_min.y"] = st.number_input("Bicarbonate (mmol/L)", 0.0, 50.0, 22.0)
        user_raw["lactate_max.y"] = st.number_input("Lactate (mmol/L)", 0.0, 30.0, 2.0)
        user_raw["anion_gap_max"] = st.number_input("Anion Gap (mmol/L)", 0.0, 50.0, 12.0)

    with st.expander("肾 Renal & Liver", expanded=True):
        user_raw["creatinine_max.y"] = st.number_input("Creatinine (mg/dL)", 0.0, 20.0, 1.0)
        user_raw["bun_max.y"] = st.number_input("BUN (mg/dL)", 0.0, 200.0, 20.0)
        user_raw["tbil_max"] = st.number_input("Total Bilirubin (mg/dL)", 0.0, 100.0, 1.0)
        user_raw["inr_max.y"] = st.number_input("INR (max)", 0.0, 15.0, 1.0)

    with st.expander("💓 Vital Signs & Others", expanded=False):
        user_raw["mbp_mean"] = st.number_input("MBP (mmHg)", 0, 200, 80)
        user_raw["heart_rate_mean"] = st.number_input("Heart Rate (bpm)", 0, 250, 80)
        user_raw["resp_rate_mean"] = st.number_input("Resp Rate (bpm)", 0, 100, 18)
        user_raw["temperature_mean"] = st.number_input("Temperature (°C)", 25.0, 45.0, 36.6)
        user_raw["platelet_min"] = st.number_input("Platelets (10^9/L)", 0, 1000, 150)
        user_raw["wbc_max.y"] = st.number_input("WBC (10^9/L)", 0.0, 200.0, 10.0)
        user_raw["pao2fio2ratio_min"] = st.number_input("PaO2/FiO2 Ratio", 0, 800, 400)

    st.markdown("---")
    predict_btn = st.button("🚀 Classify Phenotype", use_container_width=True)

# --- 4. Main Panel ---
st.title("🩺 ACLF Unsupervised Phenotype Classifier")
st.caption("Consensus Clustering (PAM) Analysis · External Validation Logic Applied")

if predict_btn:
    # 1. Z-score Transformation
    z_scores = []
    for var in SCALING.keys():
        val = (user_raw[var] - SCALING[var]["mean"]) / SCALING[var]["sd"]
        z_scores.append(val)
    
    # 2. Euclidean Distance Calculation
    dists = {}
    for name, center in CENTROIDS.items():
        dist = np.sqrt(np.sum((np.array(z_scores) - np.array(center))**2))
        dists[name] = dist
    
    matched_p = min(dists, key=dists.get)
    confidence = 1 - (min(dists.values()) / sum(dists.values()))
    
    # 3. Display Results
    color = "#2ecc71" if matched_p == "Phenotype 1" else "#e74c3c"
    st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background: white; border-left: 10px solid {color}; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h2 style="color:{color}; margin:0;">Matched: {matched_p}</h2>
            <p style="font-size:1.2em; color:#555;">Similarity Confidence: <b>{confidence:.1%}</b></p>
            <p>Based on the 15-feature multidimensional distance, this patient aligns with the <b>{matched_p}</b> clinical profile.</p>
        </div>
    """, unsafe_allow_html=True)

    # 4. Radar Chart Visualization
    categories = [SCALING[v]["label"] for v in SCALING.keys()]
    
    fig = go.Figure()
    # Phenotype 1 Baseline
    fig.add_trace(go.Scatterpolar(r=CENTROIDS["Phenotype 1"], theta=categories, fill='toself', name='Phenotype 1 (Ref)', line_color='#2ecc71', opacity=0.2))
    # Phenotype 2 Baseline
    fig.add_trace(go.Scatterpolar(r=CENTROIDS["Phenotype 2"], theta=categories, fill='toself', name='Phenotype 2 (Ref)', line_color='#e74c3c', opacity=0.2))
    # Patient Data
    fig.add_trace(go.Scatterpolar(r=z_scores, theta=categories, name='Current Patient', line=dict(color='black', width=4), marker=dict(size=8)))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2])), showlegend=True, height=600, title="Multidimensional Feature Profile (Z-Score Space)")
    st.plotly_chart(fig, use_container_width=True)
    
    

else:
    st.info("💡 **Instructions:** Enter the 15 clinical parameters from the first 24 hours of admission in the sidebar and click 'Classify Phenotype'.")