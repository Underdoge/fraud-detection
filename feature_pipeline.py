""" Feature engineering and training pipeline. """

def build_pipeline():
    """ Build pipeline with feature encoders and model. """
    import mlflow
    from sklearn.compose import ColumnTransformer  # noqa: E402
    from sklearn.ensemble import RandomForestClassifier  # noqa: E402
    from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
    from sklearn.preprocessing import RobustScaler, TargetEncoder  # noqa: E402
    # Target encoder
    internal_target_encoding = TargetEncoder(smooth="auto")
    columns_to_encode = [
        "Card",
        "Merchant Name",
        "Merchant State",
        "Merchant City",
        "Zip",
        "MCC",
        "Errors?",
        "Hour",
        "Minute",
        "Use Chip"
    ]

    mlflow.log_param("target_encoded_columns", columns_to_encode)
    encoder_params = internal_target_encoding.get_params()
    mlflow.log_params(
        {f"encoder__{key}": value for key, value in encoder_params.items()})

    target_encoding = ColumnTransformer([
        (
            'target_encode',
            internal_target_encoding,
            columns_to_encode
        )
    ])

    # Scaler
    internal_scaler = RobustScaler()
    columns_to_scale = ["Amount"]

    mlflow.log_param("scaled_columns", columns_to_scale)
    scaler_params = internal_scaler.get_params()
    mlflow.log_params(
        {f"scaler__{key}": value for key, value in scaler_params.items()})

    scaler = ColumnTransformer([
        ("scaler", internal_scaler, columns_to_scale)
    ])

    # Full pipeline
    feature_engineering_pipeline  = Pipeline([
        (
            "features",
            FeatureUnion([
                ('categories', target_encoding),
                ('scaled', scaler)
            ])
        )
    ])

    # Machine learning model
    model = RandomForestClassifier(n_estimators=10, verbose=1, n_jobs=10)

    model_params = model.get_params()
    mlflow.log_params({
        f"model__{key}": value for key, value in model_params.items()
    })

    # Full pipeline
    final_pipeline = Pipeline([
        ("feature_engineering", feature_engineering_pipeline),
        ("model", model)
    ])

    return final_pipeline
