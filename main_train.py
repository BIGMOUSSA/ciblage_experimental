import argparse
from scripts.data_processing import load_and_preprocess_data
from scripts.model_training import train_and_evaluate_model

# Define file paths and parameters
rural_train_path = "data/Train_rur_V1.1.csv"
rural_test_path = "data/Test_rur_V1.1.csv"
urban_train_path = "data/Train_urb_V1.1.csv"
urban_test_path = "data/Test_urb_V1.1.csv"

index_column = "hhid"
label_column = "pauvre"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train and evaluate models for rural or urban data.")
parser.add_argument("--milieu", choices=["rural", "urbain"], required=True, help="Specify the milieu to process (rural or urbain).")
args = parser.parse_args()

# Process and train based on the specified milieu
if args.milieu == "rural":
    print("Processing rural data...")
    rural_train, rural_test = load_and_preprocess_data(rural_train_path, rural_test_path, index_column)

    print("Training and evaluating model for rural data...")
    train_and_evaluate_model(rural_train, rural_test, label_column, output_dir="models/rural", milieu="rural")

elif args.milieu == "urbain":
    print("Processing urban data...")
    urban_train, urban_test = load_and_preprocess_data(urban_train_path, urban_test_path, index_column)

    print("Training and evaluating model for urban data...")
    train_and_evaluate_model(urban_train, urban_test, label_column, output_dir="models/urban", milieu="urban")
