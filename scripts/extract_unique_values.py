import pandas as pd

# Load a sample of the data
sample = pd.read_csv('data/Train_urb_V1.1.csv', nrows=2000)

col = [
    'partage_sanitaire', 'source_eau_ss', 'age', 'statut_occup', 'ventilo', 'evac_eau_usees',
    'eclairage', 'fer_electrique', 'nature_sol', 'materiau_toit', 'type_logement', 'type_sanitaire',
    'materiau_mur', 'voiture', 'sexe', 'mode_evac_ordure', 'fer_charbon', 'ordinateur',
    'tx_promiscuite', 'dem_emp_rate', 'activ12m', 'log_hhsize', 'bonbonne_gaz', 'frigo', 'region'
]

unique_values = {}
for c in col:
    unique_values[c] = sorted(sample[c].dropna().unique())

for k, v in unique_values.items():
    print(f'{k}: {v}')
