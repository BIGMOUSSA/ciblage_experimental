from autogluon.tabular import TabularPredictor

def train_and_evaluate_model(
    train_data,
    test_data,
    label_column,
    output_dir,
    milieu
):
    predictor = TabularPredictor(
        label=label_column,
        eval_metric="f1",
        path=output_dir
    ).fit(
        train_data,
        presets="medium_quality",
        time_limit=2*60,                 # 2 minutes max
        num_cpus=8,                     # M1
        memory_limit=6,                 # limite RAM
        num_bag_folds=0,                # énorme gain de temps
        hyperparameters={
            "GBM": {},
            "CAT": {},
        },
        verbosity=2
    )

    print(f"\n📊 Leaderboard – {milieu}")
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard)

    print(f"\n✅ Résultats sur test – {milieu}")
    results = predictor.evaluate(test_data, auxiliary_metrics=True, silent=True)
    print(results)