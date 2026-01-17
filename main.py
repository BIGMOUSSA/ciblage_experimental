from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from autogluon.tabular import TabularDataset, TabularPredictor

app = FastAPI(title="Urban Poverty Predictor API")

# Load model once on startup
predictor_V_urb = TabularPredictor.load("models/urban", require_version_match=False)
print("Model loaded successfully for urban data.")

# Define input schema using Pydantic
class InputData(BaseModel):
    log_hhsize: float
    region: int
    fer_electrique: int
    fer_charbon: int
    bonbonne_gaz: int
    ventilo: int
    voiture: int
    ordinateur: int
    frigo: int
    type_logement: int
    statut_occup: int
    eclairage: int
    materiau_toit: int
    nature_sol: int
    source_eau_ss: int
    materiau_mur: int
    tx_promiscuite: float
    dem_emp_rate: float
    activ12m: int
    age: float
    educ_hi: float
    sexe: int
    mode_evac_ordure: int
    type_sanitaire: int
    partage_sanitaire: int
    excrement_hors_concess: float
    evac_eau_usees: int
    milieu: float
    pauvre: int
    hhid: float
    source: int

@app.post("/predict")
def predict_poverty(input_data: InputData):
    input_dict = input_data.dict()
    input_df = TabularDataset([input_dict])

    prediction = predictor_V_urb.predict(input_df).map({0: "non pauvre", 1: "pauvre"})
    prediction_proba = predictor_V_urb.predict_proba(input_df).values[0][1]

    return {
        "prediction": prediction.iloc[0],
        "probability_pauvre": prediction_proba
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Urban Poverty Predictor API. Use the /predict endpoint to make predictions."}