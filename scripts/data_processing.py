import pandas as pd
from autogluon.tabular import TabularDataset
def load_and_preprocess_data(train_path, test_path, index_col):
    """
    Load and preprocess train and test datasets.

    Parameters:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the testing dataset.
        index_col (str): Column to set as index.

    Returns:
        tuple: Preprocessed train and test DataFrames.
    """
    col = [#'partage_sanitaire',
            'source_eau_ss',
            'age',
            'statut_occup',
            'ventilo',
            #'evac_eau_usees',
            'eclairage',
            'fer_electrique',
            'nature_sol',
            'materiau_toit',
            'type_logement',
            'type_sanitaire',
            'materiau_mur',
            'voiture',
            'sexe',
            #'mode_evac_ordure',
            'fer_charbon',
            'ordinateur',
            'tx_promiscuite',
            'dem_emp_rate',
            'activ12m',
            'log_hhsize',
            'bonbonne_gaz',
            'frigo',
            'region',
            'pauvre']
    # Load datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # Set index column
    train.set_index(index_col, inplace=True)
    test.set_index(index_col, inplace=True)
    # Drop unnecessary columns
    train = TabularDataset(pd.read_csv(train_path)[col])
    test = TabularDataset(pd.read_csv(test_path)[col])

    # Convert categorical columns to category dtype
    cat_cols = ['source_eau_ss', 'statut_occup', 'ventilo', 'eclairage', 'fer_electrique',
                'nature_sol', 'materiau_toit', 'type_logement', 'type_sanitaire',
                'materiau_mur', 'fer_charbon', 'ordinateur',
                'region']
    # convert to categorical data type for columns in 'rg_col'
    train[cat_cols] = train[cat_cols].astype('category')
    test[cat_cols] = test[cat_cols].astype('category')
    # Convert target variable to category dtype
    train['pauvre'] = train['pauvre'].astype('category')
    test['pauvre'] = test['pauvre'].astype('category')



    # Check for missing values
    print("Missing values in train:")
    print(train.isna().sum())
    print("\nMissing values in test:")
    print(test.isna().sum())

    return train, test
