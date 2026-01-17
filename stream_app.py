# --- Label mappings for categorical variables ---
ouinon = {1: 'oui', 2: 'non'}
activ12_cm_lab = {1: 'occupe', 2: 'non occupe'}
eclair = {1: 'Electricité SENELEC', 2: 'Générateur', 3: 'Solaire', 4: 'Lampe', 5: 'Bois', 6: "Autre Type de source d'éclairage"}
evac_llb = {1: 'puisard', 2: 'egout', 3: 'trou creuse', 4: 'dans la nature', 5: 'autre type evac'}
mat_mur = {1: 'ciment', 2: 'banco, paille, tige, motte de terre', 3: 'bois, matériaux de récupération', 4: 'autre type de revêtement du mur'}
mat_toit = {1: 'dalle en ciment, béton', 2: 'tuile, ardoise', 3: 'tole, zinc', 4: 'chaume, paille', 5: 'bois, banco', 6: 'autre type de matériau de toit'}
milieu_lbl = {1: 'Urbain', 2: 'Rural'}
nat_sol = {1: 'Carrelage, marbre', 2: 'Ciment, béton', 3: 'Terre battue, sable', 4: 'Autres types de revêtement du sol'}
region_lbl = {1: 'DAKAR', 2: 'ZIGUINCHOR', 3: 'DIOURBEL', 4: 'SAINT-LOUIS', 5: 'TAMBACOUNDA', 6: 'KAOLACK', 7: 'THIES', 8: 'LOUGA', 9: 'FATICK', 10: 'KOLDA', 11: 'MATAM', 12: 'KAFFRINE', 13: 'KEDOUGOU', 14: 'SEDHIOU'}
sexe_md = {1: 'homme', 2: 'femme'}
sour_eau = {1: 'Robinet', 2: 'Puits non protégés', 3: 'Puits protégés', 4: 'Eau de surface', 5: "Autres sources d'approvisionnement en eau"}
statut_occup_llb = {1: 'loge par employeur', 2: 'proprietaire', 3: 'coproprietaire', 4: 'locataire, colocataire', 5: 'loge gratuitement'}
typ_log_bbl = {1: 'maison a étage', 2: 'maison basse', 3: 'baraque', 4: 'case'}
type_sanitaire_llb = {1: 'chasse', 2: 'latrine', 3: 'toilette publique', 4: 'aucune toilette'}
import streamlit as st
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# Only keep these variables for inference
col = [
    'partage_sanitaire',
    'source_eau_ss',
    'age',
    'statut_occup',
    'ventilo',
    'evac_eau_usees',
    'eclairage',
    'fer_electrique',
    'nature_sol',
    'materiau_toit',
    'type_logement',
    'type_sanitaire',
    'materiau_mur',
    'voiture',
    'sexe',
    'mode_evac_ordure',
    'fer_charbon',
    'ordinateur',
    'tx_promiscuite',
    'dem_emp_rate',
    'activ12m',
    'log_hhsize',
    'bonbonne_gaz',
    'frigo',
    'region'
]

# Load the model once
@st.cache_resource
def load_model():
    return TabularPredictor.load("models/urban", require_version_match=False)

predictor = load_model()

st.title("🏙️ Urban Poverty Predictor")

# Input form
with st.form("user_input_form"):
    st.header("Enter Household Characteristics")

    log_hhsize = st.number_input("Log Household Size", value=2.6)
    region = st.selectbox("Region", [region_lbl[k] for k in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], index=6)
    fer_electrique = st.selectbox("Electric Iron", [ouinon[k] for k in [1, 2]])
    fer_charbon = st.selectbox("Charcoal Iron", [ouinon[k] for k in [1, 2]])
    bonbonne_gaz = st.selectbox("Gas Cylinder", [ouinon[k] for k in [1, 2]])
    ventilo = st.selectbox("Fan", [ouinon[k] for k in [1, 2]])
    voiture = st.selectbox("Car", [ouinon[k] for k in [1, 2]])
    ordinateur = st.selectbox("Computer", [ouinon[k] for k in [1, 2]])
    frigo = st.selectbox("Fridge", [ouinon[k] for k in [1, 2]])
    type_logement = st.selectbox("Type of Housing", [typ_log_bbl[k] for k in [1, 2, 3, 4]])
    statut_occup = st.selectbox("Occupancy Status", [statut_occup_llb[k] for k in [1, 2, 3, 4, 5]])
    eclairage = st.selectbox("Lighting", [eclair[k] for k in [1, 2, 3, 4, 5, 6]])
    materiau_toit = st.selectbox("Roof Material", [mat_toit[k] for k in [1, 2, 3, 4, 5, 6]])
    nature_sol = st.selectbox("Soil Type", [nat_sol[k] for k in [1, 2, 3, 4]])
    source_eau_ss = st.selectbox("Water Source", [sour_eau[k] for k in [1, 2, 3, 4, 5]])
    materiau_mur = st.selectbox("Wall Material", [mat_mur[k] for k in [1, 2, 3, 4]])
    tx_promiscuite = st.number_input("Promiscuity Rate", value=1.75)
    dem_emp_rate = st.number_input("Employment Demand Rate", value=7.1)
    activ12m = st.selectbox("Activity in Last 12 Months", [activ12_cm_lab[k] for k in [1, 2]])
    age = st.number_input("Age", value=60.0)
    sexe = st.selectbox("Sex", [sexe_md[k] for k in [1, 2]])
    mode_evac_ordure = st.selectbox("Waste Disposal Method", [1, 2, 3, 4, 5, 6])  # No label mapping provided
    type_sanitaire = st.selectbox("Sanitary Type", [type_sanitaire_llb[k] for k in [1, 2, 3, 4]])
    partage_sanitaire = st.selectbox("Shared Toilet", [ouinon[k] for k in [1, 2]])
    evac_eau_usees = st.selectbox("Wastewater Evacuation", [evac_llb[k] for k in [1, 2, 3, 4, 5]])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build input_dict with only the required columns for inference
    # Reverse mapping from label to code for each categorical variable
    region_inv = {v: k for k, v in region_lbl.items()}
    ouinon_inv = {v: k for k, v in ouinon.items()}
    activ12_cm_lab_inv = {v: k for k, v in activ12_cm_lab.items()}
    eclair_inv = {v: k for k, v in eclair.items()}
    evac_llb_inv = {v: k for k, v in evac_llb.items()}
    mat_mur_inv = {v: k for k, v in mat_mur.items()}
    mat_toit_inv = {v: k for k, v in mat_toit.items()}
    nat_sol_inv = {v: k for k, v in nat_sol.items()}
    sexe_md_inv = {v: k for k, v in sexe_md.items()}
    sour_eau_inv = {v: k for k, v in sour_eau.items()}
    statut_occup_llb_inv = {v: k for k, v in statut_occup_llb.items()}
    typ_log_bbl_inv = {v: k for k, v in typ_log_bbl.items()}
    type_sanitaire_llb_inv = {v: k for k, v in type_sanitaire_llb.items()}

    input_dict = {
        'partage_sanitaire': ouinon_inv[partage_sanitaire],
        'source_eau_ss': sour_eau_inv[source_eau_ss],
        'age': age,
        'statut_occup': statut_occup_llb_inv[statut_occup],
        'ventilo': ouinon_inv[ventilo],
        'evac_eau_usees': evac_llb_inv[evac_eau_usees],
        'eclairage': eclair_inv[eclairage],
        'fer_electrique': ouinon_inv[fer_electrique],
        'nature_sol': nat_sol_inv[nature_sol],
        'materiau_toit': mat_toit_inv[materiau_toit],
        'type_logement': typ_log_bbl_inv[type_logement],
        'type_sanitaire': type_sanitaire_llb_inv[type_sanitaire],
        'materiau_mur': mat_mur_inv[materiau_mur],
        'voiture': ouinon_inv[voiture],
        'sexe': sexe_md_inv[sexe],
        'mode_evac_ordure': mode_evac_ordure,  # No label mapping provided
        'fer_charbon': ouinon_inv[fer_charbon],
        'ordinateur': ouinon_inv[ordinateur],
        'tx_promiscuite': tx_promiscuite,
        'dem_emp_rate': dem_emp_rate,
        'activ12m': activ12_cm_lab_inv[activ12m],
        'log_hhsize': log_hhsize,
        'bonbonne_gaz': ouinon_inv[bonbonne_gaz],
        'frigo': ouinon_inv[frigo],
        'region': region_inv[region]
    }

    input_df = TabularDataset([input_dict])
    prediction = predictor.predict(input_df).map({0: "non pauvre", 1: "pauvre"})
    prediction_proba = predictor.predict_proba(input_df).values[0][1]

    st.success(f"🧠 Prediction: **{prediction.iloc[0]}**") #
    st.info(f"📊 Probability of being 'pauvre': **{prediction_proba:.2%}**")
