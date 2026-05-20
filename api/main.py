from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "models/xgboost_model.pkl"
)

model_features = joblib.load(
    "models/model_features.pkl"
)

# =========================
# CREATE FASTAPI APP
# =========================

app = FastAPI(
    title="Disaster Recovery Cost Prediction API"
)

# =========================
# REQUEST SCHEMA
# =========================

class PredictionRequest(BaseModel):

    incident_duration_days: int

    declaration_year: int

    declaration_month: int

    state: str

    incidentType: str

    declarationType: str

# =========================
# HEALTH ENDPOINT
# =========================

@app.get("/health")

def health_check():

    return {
        "status": "API is running"
    }

# =========================
# PREDICTION ENDPOINT
# =========================

@app.post("/predict-cost")

def predict_cost(request: PredictionRequest):

    input_data = {}

    # initialize all features to 0

    for feature in model_features:

        input_data[feature] = 0

    # numerical features

    input_data["incident_duration_days"] = (
        request.incident_duration_days
    )

    input_data["declaration_year"] = (
        request.declaration_year
    )

    input_data["declaration_month"] = (
        request.declaration_month
    )

    # categorical encoding

    state_feature = f"state_{request.state}"

    incident_feature = (
        f"incidentType_{request.incidentType}"
    )

    declaration_feature = (
        f"declarationType_{request.declarationType}"
    )

    if state_feature in input_data:

        input_data[state_feature] = 1

    if incident_feature in input_data:

        input_data[incident_feature] = 1

    if declaration_feature in input_data:

        input_data[declaration_feature] = 1

    # create dataframe

    input_df = pd.DataFrame(
        [input_data]
    )

    # reorder columns

    input_df = input_df[
        model_features
    ]

    # prediction

    prediction_log = model.predict(
        input_df
    )[0]

    prediction = np.expm1(
        prediction_log
    )

    return {
        "predicted_recovery_cost": round(
            float(prediction),
            2
        )
    }