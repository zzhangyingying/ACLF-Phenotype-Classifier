"""Streamlit interface for research-only ACLF phenotype template matching."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import json
from typing import Any, Mapping


st.set_page_config(
    page_title="First-day ACLF Phenotype Research Tool",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .research-box {
        padding: 1.1rem 1.3rem;
        border-radius: 0.65rem;
        border-left: 0.45rem solid #3C5488;
        background: #F7F8FB;
        margin: 0.6rem 0 1rem 0;
    }
    .result-box {
        padding: 1.25rem 1.45rem;
        border-radius: 0.75rem;
        background: white;
        border-left: 0.6rem solid;
        box-shadow: 0 3px 14px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .small-note {font-size: 0.88rem; color: #555; line-height: 1.45;}
    </style>
    """,
    unsafe_allow_html=True,
)


def classify_complete_inputs(
    values: Mapping[str, float | int | None],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign a research phenotype when all 15 required inputs are available.

    Missing values are never imputed. Values outside the versioned plausibility
    bounds are rejected to avoid undocumented extrapolation.
    """
    features = model["features"]
    missing = [
        feature["key"]
        for feature in features
        if values.get(feature["key"]) is None
    ]
    if missing:
        raise ValueError("Missing required inputs: " + ", ".join(missing))

    raw = np.asarray([float(values[feature["key"]]) for feature in features])
    if not np.all(np.isfinite(raw)):
        raise ValueError("All 15 inputs must be finite numeric values.")

    outside = []
    for value, feature in zip(raw, features):
        lower = float(feature["plausibility_min"])
        upper = float(feature["plausibility_max"])
        if value < lower or value > upper:
            outside.append(
                f"{feature['label']} ({value:g}; expected {lower:g} to {upper:g} {feature['unit']})"
            )
    if outside:
        raise ValueError("Inputs outside the permitted research-entry range: " + "; ".join(outside))

    center = np.asarray([float(feature["center"]) for feature in features])
    scale = np.asarray([float(feature["scale"]) for feature in features])
    centroid_1 = np.asarray([float(feature["centroid_1"]) for feature in features])
    centroid_2 = np.asarray([float(feature["centroid_2"]) for feature in features])

    standardized = (raw - center) / scale
    distance_1 = float(np.linalg.norm(standardized - centroid_1))
    distance_2 = float(np.linalg.norm(standardized - centroid_2))
    denominator = distance_1 + distance_2
    normalized_margin = 0.0 if denominator == 0 else abs(distance_1 - distance_2) / denominator
    assignment = "Phenotype 1" if distance_1 <= distance_2 else "Phenotype 2"
    threshold = float(model["borderline_threshold"])

    return {
        "assignment": assignment,
        "distance_to_phenotype_1": distance_1,
        "distance_to_phenotype_2": distance_2,
        "normalized_margin": normalized_margin,
        "borderline": normalized_margin < threshold,
        "borderline_threshold": threshold,
        "standardized_values": standardized.tolist(),
    }



MODEL_JSON = r'''
{
  "model_name": "First-day ACLF phenotype research assignment tool",
  "model_version": "1.0.0",
  "algorithm_release_date": "2026-07-19",
  "source_database": "MIMIC-IV v3.1",
  "derivation_cohort_n": 1845,
  "feature_window": "First 24 hours after ICU admission",
  "algorithm": "Nearest-centroid template matching in the derivation-cohort standardized 15-feature space",
  "borderline_metric": "abs(d1-d2)/(d1+d2)",
  "borderline_threshold": 0.0277733493734055,
  "borderline_threshold_source": "10th percentile of the normalized distance margin in the MIMIC-IV derivation cohort",
  "missing_input_policy": "All 15 inputs are required; no automatic imputation or default-value substitution is performed",
  "features": [
    {
      "key": "ph_min_24h",
      "label": "pH",
      "short_label": "pH (min)",
      "unit": "unitless",
      "aggregation": "Minimum during first 24 h",
      "group": "Metabolic and acid-base",
      "plausibility_min": 6.5,
      "plausibility_max": 8.0,
      "step": 0.01,
      "format": "%.2f",
      "center": 7.3172891598916,
      "scale": 0.11465977497947,
      "centroid_1": 0.417608473846709,
      "centroid_2": -0.630674021727682
    },
    {
      "key": "bicarbonate_min_24h",
      "label": "Bicarbonate",
      "short_label": "Bicarbonate (min)",
      "unit": "mEq/L",
      "aggregation": "Minimum during first 24 h",
      "group": "Metabolic and acid-base",
      "plausibility_min": 2.0,
      "plausibility_max": 60.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 19.1038482384824,
      "scale": 5.31750753500731,
      "centroid_1": 0.568601059687909,
      "centroid_2": -0.858703641161332
    },
    {
      "key": "lactate_max_24h",
      "label": "Lactate",
      "short_label": "Lactate (max)",
      "unit": "mmol/L",
      "aggregation": "Maximum during first 24 h",
      "group": "Metabolic and acid-base",
      "plausibility_min": 0.1,
      "plausibility_max": 40.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 3.87210027100271,
      "scale": 3.77878214366614,
      "centroid_1": -0.39098507443176,
      "centroid_2": 0.59046725526429
    },
    {
      "key": "anion_gap_max_24h",
      "label": "Anion gap",
      "short_label": "Anion gap (max)",
      "unit": "mEq/L",
      "aggregation": "Maximum during first 24 h",
      "group": "Metabolic and acid-base",
      "plausibility_min": 0.0,
      "plausibility_max": 60.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 18.360243902439,
      "scale": 6.38037830246728,
      "centroid_1": -0.517806610309354,
      "centroid_2": 0.781993656385555
    },
    {
      "key": "map_mean_24h",
      "label": "Mean arterial pressure",
      "short_label": "MAP (mean)",
      "unit": "mmHg",
      "aggregation": "Mean during first 24 h",
      "group": "Vital signs",
      "plausibility_min": 20.0,
      "plausibility_max": 200.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 73.6196639523035,
      "scale": 9.48675489447391,
      "centroid_1": 0.0882367481428886,
      "centroid_2": -0.133255497195383
    },
    {
      "key": "heart_rate_mean_24h",
      "label": "Heart rate",
      "short_label": "Heart rate (mean)",
      "unit": "beats/min",
      "aggregation": "Mean during first 24 h",
      "group": "Vital signs",
      "plausibility_min": 20.0,
      "plausibility_max": 250.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 89.0174222142536,
      "scale": 16.4905946805481,
      "centroid_1": -0.212717887267701,
      "centroid_2": 0.321247421587956
    },
    {
      "key": "respiratory_rate_mean_24h",
      "label": "Respiratory rate",
      "short_label": "Respiratory rate (mean)",
      "unit": "breaths/min",
      "aggregation": "Mean during first 24 h",
      "group": "Vital signs",
      "plausibility_min": 2.0,
      "plausibility_max": 80.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 19.5771212935396,
      "scale": 4.40533014352866,
      "centroid_1": -0.256503392053761,
      "centroid_2": 0.38737246963221
    },
    {
      "key": "creatinine_max_24h",
      "label": "Creatinine",
      "short_label": "Creatinine (max)",
      "unit": "mg/dL",
      "aggregation": "Maximum during first 24 h",
      "group": "Renal and liver",
      "plausibility_min": 0.1,
      "plausibility_max": 25.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 2.49627371273713,
      "scale": 2.07209838475337,
      "centroid_1": -0.281421752610425,
      "centroid_2": 0.425004279452479
    },
    {
      "key": "bun_max_24h",
      "label": "Blood urea nitrogen",
      "short_label": "BUN (max)",
      "unit": "mg/dL",
      "aggregation": "Maximum during first 24 h",
      "group": "Renal and liver",
      "plausibility_min": 1.0,
      "plausibility_max": 300.0,
      "step": 1.0,
      "format": "%.0f",
      "center": 45.3071544715447,
      "scale": 31.9649043049475,
      "centroid_1": -0.221097136298752,
      "centroid_2": 0.333901797675666
    },
    {
      "key": "bilirubin_max_24h",
      "label": "Total bilirubin",
      "short_label": "Total bilirubin (max)",
      "unit": "mg/dL",
      "aggregation": "Maximum during first 24 h",
      "group": "Renal and liver",
      "plausibility_min": 0.1,
      "plausibility_max": 80.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 7.76522764227642,
      "scale": 9.94054174300399,
      "centroid_1": -0.178811404776572,
      "centroid_2": 0.270041713336048
    },
    {
      "key": "inr_max_24h",
      "label": "INR",
      "short_label": "INR (max)",
      "unit": "unitless",
      "aggregation": "Maximum during first 24 h",
      "group": "Renal and liver",
      "plausibility_min": 0.5,
      "plausibility_max": 25.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 2.19111924119241,
      "scale": 1.15524182525494,
      "centroid_1": -0.258498138122117,
      "centroid_2": 0.390384943286462
    },
    {
      "key": "platelet_min_24h",
      "label": "Platelet count",
      "short_label": "Platelet count (min)",
      "unit": "x10^9/L",
      "aggregation": "Minimum during first 24 h",
      "group": "Hematology and respiratory",
      "plausibility_min": 1.0,
      "plausibility_max": 2000.0,
      "step": 1.0,
      "format": "%.0f",
      "center": 113.208536585366,
      "scale": 85.430109257807,
      "centroid_1": 0.108061143193859,
      "centroid_2": -0.163194379517256
    },
    {
      "key": "wbc_max_24h",
      "label": "White blood cell count",
      "short_label": "WBC (max)",
      "unit": "x10^9/L",
      "aggregation": "Maximum during first 24 h",
      "group": "Hematology and respiratory",
      "plausibility_min": 0.1,
      "plausibility_max": 300.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 15.1250216802168,
      "scale": 11.0970135811587,
      "centroid_1": -0.197591468552983,
      "centroid_2": 0.298403442304505
    },
    {
      "key": "temperature_mean_24h",
      "label": "Temperature",
      "short_label": "Temperature (mean)",
      "unit": "degrees C",
      "aggregation": "Mean during first 24 h",
      "group": "Vital signs",
      "plausibility_min": 25.0,
      "plausibility_max": 45.0,
      "step": 0.1,
      "format": "%.1f",
      "center": 36.7588029430907,
      "scale": 0.536165298416121,
      "centroid_1": 0.112218178459833,
      "centroid_2": -0.169472351143421
    },
    {
      "key": "oxygenation_pf_min_24h",
      "label": "PaO2/FiO2",
      "short_label": "PaO2/FiO2 (min)",
      "unit": "ratio",
      "aggregation": "Minimum during first 24 h",
      "group": "Hematology and respiratory",
      "plausibility_min": 20.0,
      "plausibility_max": 800.0,
      "step": 1.0,
      "format": "%.0f",
      "center": 169.113982488063,
      "scale": 108.21106628986,
      "centroid_1": 0.169720565163255,
      "centroid_2": -0.256312690246548
    }
  ]
}
'''
model = json.loads(MODEL_JSON)
features = model["features"]

st.title("First-day ACLF Phenotype Research Assignment Tool")
st.caption(
    "Research-only nearest-template matching based on 15 physiologic and laboratory "
    "variables summarized during the first 24 hours after ICU admission."
)
st.warning(
    "Research use only. This application has not been validated for diagnosis, "
    "treatment selection, prognostic clinical decisions, or other clinical "
    "decision-support purposes. It must not replace professional clinical judgment."
)

st.markdown(
    f"""
    <div class="research-box">
      <b>Algorithm version:</b> {model['model_version']} &nbsp;|&nbsp;
      <b>Release date:</b> {model['algorithm_release_date']} &nbsp;|&nbsp;
      <b>Derivation source:</b> {model['source_database']} (n={model['derivation_cohort_n']:,})<br>
      <span class="small-note">
        All 15 inputs are required. The application performs no automatic imputation,
        default-value substitution, or outcome prediction.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

group_order = [
    "Metabolic and acid-base",
    "Renal and liver",
    "Vital signs",
    "Hematology and respiratory",
]
features_by_group = OrderedDict(
    (group, [feature for feature in features if feature["group"] == group])
    for group in group_order
)

user_values: dict[str, float | None] = {}
with st.sidebar:
    st.header("First-day clinical inputs")
    st.info(
        "Enter values summarized during the first 24 hours after ICU admission. "
        "Blank fields remain missing and are never replaced automatically."
    )

    for group, group_features in features_by_group.items():
        with st.expander(group, expanded=True):
            for feature in group_features:
                label = (
                    f"{feature['label']} — {feature['aggregation']} "
                    f"[{feature['unit']}]"
                )
                user_values[feature["key"]] = st.number_input(
                    label,
                    min_value=float(feature["plausibility_min"]),
                    max_value=float(feature["plausibility_max"]),
                    value=None,
                    step=float(feature["step"]),
                    format=feature["format"],
                    placeholder="Required",
                    key=f"input_{feature['key']}",
                    help=(
                        f"Permitted research-entry range: "
                        f"{feature['plausibility_min']:g}–{feature['plausibility_max']:g} "
                        f"{feature['unit']}. This is a plausibility screen, not a clinical threshold."
                    ),
                )

    missing_features = [
        feature for feature in features if user_values[feature["key"]] is None
    ]
    st.markdown("---")
    if missing_features:
        st.caption(f"Required inputs remaining: {len(missing_features)} of 15")
        with st.expander("View missing inputs"):
            for feature in missing_features:
                st.write(f"• {feature['label']} ({feature['unit']})")
    else:
        st.success("All 15 required inputs are complete.")

    assign_button = st.button(
        "Generate research phenotype assignment",
        use_container_width=True,
        type="primary",
        disabled=bool(missing_features),
    )


def add_radar_trace(
    figure: go.Figure,
    values: list[float],
    labels: list[str],
    name: str,
    color: str,
    fill: str | None,
    width: float,
    opacity: float,
) -> None:
    figure.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            mode="lines+markers",
            fill=fill,
            name=name,
            line=dict(color=color, width=width),
            marker=dict(color=color, size=6),
            opacity=opacity,
            hovertemplate="%{theta}: %{r:.2f} z<extra>%{fullData.name}</extra>",
        )
    )


if assign_button:
    try:
        result = classify_complete_inputs(user_values, model)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    assignment = result["assignment"]
    assignment_color = "#3C5488" if assignment == "Phenotype 1" else "#E64B35"
    borderline_text = (
        "Borderline assignment: the two template distances were similar."
        if result["borderline"]
        else "Non-borderline assignment under the prespecified derivation-cohort threshold."
    )
    st.markdown(
        f"""
        <div class="result-box" style="border-left-color: {assignment_color};">
          <h2 style="color:{assignment_color}; margin:0 0 0.35rem 0;">
            Research assignment: {assignment}
          </h2>
          <p style="margin:0.15rem 0;"><b>{borderline_text}</b></p>
          <p class="small-note" style="margin-bottom:0;">
            This is a descriptive nearest-template assignment, not a diagnostic,
            treatment-selection, or prognostic probability.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Distance to Phenotype 1", f"{result['distance_to_phenotype_1']:.3f}")
    metric_2.metric("Distance to Phenotype 2", f"{result['distance_to_phenotype_2']:.3f}")
    metric_3.metric(
        "Normalized distance margin",
        f"{result['normalized_margin']:.3f}",
        help=(
            "|d1-d2|/(d1+d2). Values closer to zero indicate greater assignment ambiguity. "
            f"The prespecified borderline threshold is {result['borderline_threshold']:.4f}."
        ),
    )

    labels = [feature["short_label"] for feature in features]
    centroid_1 = [float(feature["centroid_1"]) for feature in features]
    centroid_2 = [float(feature["centroid_2"]) for feature in features]
    patient_z = [float(value) for value in result["standardized_values"]]
    radial_limit = max(
        3.0,
        float(np.ceil(max(abs(value) for value in centroid_1 + centroid_2 + patient_z))),
    )

    figure = go.Figure()
    add_radar_trace(
        figure, centroid_1, labels, "Phenotype 1 reference", "#3C5488", "toself", 2.0, 0.34
    )
    add_radar_trace(
        figure, centroid_2, labels, "Phenotype 2 reference", "#E64B35", "toself", 2.0, 0.34
    )
    add_radar_trace(
        figure, patient_z, labels, "Current input profile", "#111111", None, 3.5, 1.0
    )
    figure.update_layout(
        title="Standardized 15-variable profile relative to the reference templates",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-radial_limit, radial_limit],
                gridcolor="#D8D8D8",
            ),
            angularaxis=dict(gridcolor="#E2E2E2"),
        ),
        showlegend=True,
        height=690,
        margin=dict(t=85, b=45, l=55, r=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    st.plotly_chart(figure, use_container_width=True)
else:
    st.info(
        "Complete all 15 required inputs in the sidebar to enable research phenotype assignment."
    )

st.markdown("---")
st.caption(
    "Research-use disclaimer: The application does not diagnose ACLF, estimate an individual "
    "patient's mortality risk, recommend treatment, or replace clinical judgment. Do not enter "
    "direct patient identifiers. This application code does not intentionally persist entered values."
)

