from autogluon.tabular import TabularDataset, TabularPredictor
# Load the test urbain dataset
df_urbain = TabularDataset("data/Test_urb_V1.1.csv")

predictor_V_urb = TabularPredictor.load("models/urban", require_version_match=False)
#result = predictor_V_urb.evaluate(df_urbain, decision_threshold=0.5)

#print("Evaluation completed for urban dataset.", result)

input = input_data = {
    "log_hhsize": 2.6390573296152584,
    "region": 3,
    "fer_electrique": 2,
    "fer_charbon": 2,
    "bonbonne_gaz": 2,
    "ventilo": 1,
    "voiture": 2,
    "ordinateur": 2,
    "frigo": 2,
    "type_logement": 2,
    "statut_occup": 2,
    "eclairage": 1,
    "materiau_toit": 3,
    "nature_sol": 2,
    "source_eau_ss": 5,
    "materiau_mur": 1,
    "tx_promiscuite": 1.75,
    "dem_emp_rate": 7.142857142857142,
    "activ12m": 1,
    "age": 60.0,
    "educ_hi": 1.0,
    "sexe": 1,
    "mode_evac_ordure": 2,
    "type_sanitaire": 1,
    "partage_sanitaire": 2,
    "excrement_hors_concess": 1.0,
    "evac_eau_usees": 4,
    "milieu": 1.0,
    "pauvre": 1,
    "hhid": 14002.0,
    "source": 1
}


# transformed into tabular dataset
input_df = TabularDataset([input])
# Make a prediction
prediction = predictor_V_urb.predict(input_df)  

# rename prediction if 0 :"non pauvre" if 1 :"pauvre"
prediction = prediction.map({0: "non pauvre", 1: "pauvre"}) 
# Display the input data and prediction
print("Model loaded successfully for urban data.")

print("Prediction for the input data:", prediction.iloc[0])

prediction_proba = predictor_V_urb.predict_proba(input_df)
print("Prediction probabilities for the input data:", prediction_proba.values[0][1])

