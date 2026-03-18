import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. Authentic 15-Feature Parameters from your R Analysis ---
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

# Centroids in Z-score space (15 dimensions)
CENTROIDS = {
    "Phenotype 1": [0.450, 0.506, -0.433, -0.527, 0.0868, -0.354, -0.349, -0.252, -0.224, -0.052, -0.165, 0.063, -0.075, 0.045, 0.125],
    "Phenotype 2": [-0.574, -0.645, 0.553, 0.673, -0.111, 0.451, 0.445, 0.321, 0.286, 0.066, 0.210, -0.081, 0.096, -0.057, -0.160]
}

# --- 2. Page UI ---
st.set_page_config(page_title="ACLF Phenotype Classifier", layout="wide")
st.markdown("""
    <style>
    .reportview-container { background: #fafafa; }
    .stNumberInput label { font-weight: bold; color: #333; }
    .result-box { padding: 25px; border-radius: 15px; background: white; border-left: 12px solid; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Sidebar: Input Grid (All 15 Variables) ---
with st.sidebar:
    st.header("📋 Patient Clinical Data")
    st.info("Please fill in all 15 parameters recorded at admission.")
    
    user_vals = {}
    
    # Group 1: Metabolic & Acid-Base
    with st.expander("🧪 Metabolic & Acid-Base", expanded=True):
        user_vals["ph_min.y"] = st.number_input("pH (min)", 6.5, 8.0, 7.30, step=0.01, format="%.2f")
        user_vals["bicarbonate_min.y"] = st.number_input("Bicarbonate (mmol/L)", 0.0, 50.0, 18.0)
        user_vals["lactate_max.y"] = st.number_input("Lactate (mmol/L)", 0.0, 30.0, 4.0)
        user_vals["anion_gap_max"] = st.number_input("Anion Gap (mmol/L)", 0.0, 50.0, 15.0)

    # Group 2: Renal & Liver
    with st.expander("肾 Renal & Liver", expanded=True):
        user_vals["creatinine_max.y"] = st.number_input("Creatinine (mg/dL)", 0.0, 20.0, 1.5)
        user_vals["bun_max.y"] = st.number_input("BUN (mg/dL)", 0.0, 200.0, 40.0)
        user_vals["tbil_max"] = st.number_input("Total Bilirubin (mg/dL)", 0.0, 100.0, 5.0)
        user_vals["inr_max.y"] = st.number_input("INR (max)", 0.0, 15.0, 1.5)

    # Group 3: Vital Signs
    with st.expander("💓 Vital Signs", expanded=True):
        user_vals["mbp_mean"] = st.number_input("MBP (mmHg)", 0, 200, 75)
        user_vals["heart_rate_mean"] = st.number_input("Heart Rate (bpm)", 0, 250, 90)
        user_vals["resp_rate_mean"] = st.number_input("Resp Rate (bpm)", 0, 100, 20)
        user_vals["temperature_mean"] = st.number_input("Temp (°C)", 25.0, 45.0, 36.5)

    # Group 4: Hematology & Respiratory
    with st.expander("🧬 Hematology & Resp", expanded=True):
        user_vals["platelet_min"] = st.number_input("Platelets (10^9/L)", 0, 1000, 100)
        user_vals["wbc_max.y"] = st.number_input("WBC (10^9/L)", 0.0, 200.0, 12.0)
        user_vals["pao2fio2ratio_min"] = st.number_input("PaO2/FiO2 Ratio", 0, 800, 250)

    st.markdown("---")
    predict_btn = st.button("🚀 Match Clinical Phenotype", use_container_width=True)

# --- 4. Prediction Logic ---
st.title("🏥 ACLF Baseline Phenotype Trajectory Matcher")
st.caption("Multidimensional Classification based on 15 Clinical Features")

if predict_btn:
    # 1. Standardize user inputs (Z-score)
    user_z = []
    for var in SCALING.keys():
        z = (user_vals[var] - SCALING[var]["mean"]) / SCALING[var]["sd"]
        user_z.append(z)
    
    # 2. Calculate Euclidean Distance to both Centroids
    dists = {}
    for name, center in CENTROIDS.items():
        d = np.sqrt(np.sum((np.array(user_z) - np.array(center))**2))
        dists[name] = d
    
    best_match = min(dists, key=dists.get)
    confidence = 1 - (min(dists.values()) / sum(dists.values()))
    
    # 3. Visual Result Box
    res_color = "#2ecc71" if best_match == "Phenotype 1" else "#e67e22"
    st.markdown(f"""
        <div class="result-box" style="border-left-color: {res_color};">
            <h2 style="color: {res_color}; margin:0;">Matched: {best_match}</h2>
            <p style="font-size: 1.2em; color: #555;">Classification Confidence: <b>{confidence:.1%}</b></p>
            <p style="margin-bottom:0;">The patient's 15-dimensional clinical profile aligns closest with <b>{best_match}</b>.</p>
        </div>
    """, unsafe_allow_html=True)

    # 4. Multidimensional Radar Chart
    labels = [SCALING[v]["label"] for v in SCALING.keys()]
    
    fig = go.Figure()
    # Baseline Phenotype 1
    fig.add_trace(go.Scatterpolar(r=CENTROIDS["Phenotype 1"], theta=labels, fill='toself', name='Phenotype 1 Ref', line_color='#2ecc71', opacity=0.2))
    # Baseline Phenotype 2
    fig.add_trace(go.Scatterpolar(r=CENTROIDS["Phenotype 2"], theta=labels, fill='toself', name='Phenotype 2 Ref', line_color='#e67e22', opacity=0.2))
    # Current Patient
    fig.add_trace(go.Scatterpolar(r=user_z, theta=labels, name='Current Patient', line=dict(color='black', width=4), marker=dict(size=8, color='black')))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2.5, 2.5])),
        showlegend=True, height=650, margin=dict(t=80, b=40),
        title="15-Dimensional Clinical Fingerprint (Z-Score Space)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.info("💡 **Instructions:** Enter all 15 clinical parameters in the sidebar. The radar chart will visualize the patient's 'clinical fingerprint' compared to the two major ACLF phenotypes.")