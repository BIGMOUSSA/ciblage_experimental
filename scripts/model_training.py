from autogluon.tabular import TabularPredictor

def train_and_evaluate_model(train_data, test_data, label_column, output_dir, milieu, result):
    """
    Train and evaluate a model using AutoGluon with optimization and a progress bar.

    Parameters:
        train_data (DataFrame): Training dataset.
        test_data (DataFrame): Testing dataset.
        label_column (str): Name of the target column.
        output_dir (str): Directory to save the trained model.

    Returns:
        None
    """
    # Initialize the predictor with optimization settings
        # Train the model with hyperparameter optimization and progress tracking
    hyperparameters={
        "RF": [
            {}  # RandomForestEntr
        ],
        "XT": [
            {}                         # ExtraTrees (meilleur sera sélectionné)
        ],
        "CAT": { }
    }

    predictor = TabularPredictor(label=label_column, eval_metric="f1", path=output_dir).fit(train_data,presets="best_quality",time_limit= 1*60, hyperparameters=hyperparameters, verbosity=3)

        # Evaluate the model
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(f"Model Leaderboard {milieu}:")
    print(leaderboard)
    print(f"resultats model {milieu} on test set")
    results = predictor.evaluate(test_data, auxiliary_metrics=True, silent=True)
    print(f" {milieu} :  {results}")
    # save results to a file
    import os

    os.makedirs(result, exist_ok=True)

    results_path = os.path.join(result, f"results_{milieu}.txt")

    with open(results_path, "w") as f:
        f.write(f"Model Leaderboard {milieu}:\n")
        f.write(leaderboard.to_string())
        f.write("\n\nResults on test set:\n")
        f.write(str(results))

    print(f"✅ Results written to {results_path}")
        

