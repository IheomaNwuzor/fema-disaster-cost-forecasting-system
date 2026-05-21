import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "https://fema-disaster-cost-forecasting-system.onrender.com/predict-cost"

st.set_page_config(
    page_title="FEMA Disaster Recovery Cost Forecasting System",
    layout="wide"
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 320px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("FEMA Disaster Recovery Cost Forecasting System")

st.write(
    "Forecast FEMA disaster recovery costs using a trained XGBoost model."
)

st.caption(
    "Predictions are estimates based on historical FEMA disaster recovery data, not guaranteed final costs."
)

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Disaster Scenario Inputs")

incident_duration_days = st.sidebar.number_input(
    "Incident Duration (Days)",
    min_value=0,
    value=10
)

declaration_year = st.sidebar.number_input(
    "Declaration Year",
    min_value=2000,
    max_value=2035,
    value=2024
)

declaration_month = st.sidebar.selectbox(
    "Declaration Month",
    options=list(range(1, 13)),
    index=7
)

state = st.sidebar.selectbox(
    "State",
    ["FL", "CA", "TX", "NY", "LA", "AL", "PR", "MT", "SD", "ME"]
)

incident_type = st.sidebar.selectbox(
    "Incident Type",
    [
        "Hurricane",
        "Flood",
        "Fire",
        "Severe Storm",
        "Snowstorm",
        "Biological",
        "Severe Ice Storm",
        "Winter Storm",
        "Tornado"
    ]
)

declaration_type = st.sidebar.selectbox(
    "Declaration Type",
    ["DR", "EM"]
)

budget_available = st.sidebar.number_input(
    "Available Budget ($)",
    min_value=0.0,
    value=1_000_000.0,
    step=100_000.0
)

# =========================
# MAIN PREDICTION
# =========================

payload = {
    "incident_duration_days": incident_duration_days,
    "declaration_year": declaration_year,
    "declaration_month": declaration_month,
    "state": state,
    "incidentType": incident_type,
    "declarationType": declaration_type
}

if st.button("Predict Recovery Cost"):

    response = requests.post(
        API_URL,
        json=payload
    )

    if response.status_code == 200:

        prediction = response.json()["predicted_recovery_cost"]

        budget_gap = prediction - budget_available

        st.divider()

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Recovery Cost",
            f"${prediction:,.2f}"
        )

        col2.metric(
            "Available Budget",
            f"${budget_available:,.2f}"
        )

        col3.metric(
            "Budget Gap",
            f"${budget_gap:,.2f}"
        )

        # =========================
        # BUDGET STATUS
        # =========================

        if budget_gap > 0:

            st.error(
                "Forecasted recovery cost exceeds available budget."
            )

        else:

            st.success(
                "Available budget is sufficient for the predicted cost."
            )

        # =========================
        # DISASTER SEVERITY
        # =========================

        if prediction < 10_000_000:

            st.info(
                "Low Impact Disaster Forecast"
            )

        elif prediction < 100_000_000:

            st.warning(
                "Moderate Impact Disaster Forecast"
            )

        else:

            st.error(
                "Severe High-Cost Disaster Forecast"
            )

        st.caption(
            "This forecast should support planning decisions, not replace expert disaster finance review."
        )

    else:

        st.error(
            "Prediction failed. Check if FastAPI backend is running."
        )

# =========================
# SCENARIO SIMULATION
# =========================

st.divider()

st.subheader("Scenario Simulation")

st.write(
    "Compare how predicted recovery cost changes under different disaster durations."
)

scenario_durations = [5, 10, 20, 30, 60, 120]

scenario_results = []

for duration in scenario_durations:

    scenario_payload = payload.copy()

    scenario_payload["incident_duration_days"] = duration

    response = requests.post(
        API_URL,
        json=scenario_payload
    )

    if response.status_code == 200:

        predicted_cost = response.json()["predicted_recovery_cost"]

        scenario_results.append(
            {
                "Incident Duration Days": duration,
                "Predicted Recovery Cost": predicted_cost
            }
        )

if scenario_results:

    scenario_chart_df = pd.DataFrame(scenario_results)

    scenario_table_df = scenario_chart_df.copy()

    scenario_table_df["Predicted Recovery Cost"] = scenario_table_df[
        "Predicted Recovery Cost"
    ].apply(
        lambda x: f"${x:,.2f}"
    )

    st.dataframe(
        scenario_table_df,
        use_container_width=True
    )

    fig = px.line(
        scenario_chart_df,
        x="Incident Duration Days",
        y="Predicted Recovery Cost",
        markers=True,
        title="Scenario Forecast Trend"
    )

    fig.update_layout(
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Scenario results may not increase smoothly because the model is learning from historical disaster patterns, including outliers."
    )

# =========================
# MODEL PERFORMANCE
# =========================

st.divider()

st.subheader("Model Performance")

st.write(
    """
    Final selected model: **XGBoost Regressor**

    - R² Score: **0.52**
    - MAE: **1.04**
    - RMSE: **1.44**

    The model explains approximately **52% of the variation** in disaster recovery costs using the current FEMA-based features.
    """
)

# =========================
# FEATURE IMPORTANCE
# =========================

st.divider()

st.subheader("Model Feature Importance")

st.write(
    "The chart below shows which features most influenced the XGBoost model."
)

st.image(
    "reports/xgboost_feature_importance.png",
    width=1000
)