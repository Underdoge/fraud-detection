""" Methods to build the model's feature engineering and training pipeline. """

from numpy import array
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder, RobustScaler


def build_pipeline() -> Pipeline:
    """ Creates feature engineering and traiing pipeline.

    Returns:
        final_pipeline (Pipeline): The final pipeline.
    """

    # Binarizer
    internal_binarizer = Binarizer()
    columns_to_binarize = ["Won"]

    internal_encoder_binarizer = OneHotEncoder(sparse_output=False,
                                               handle_unknown="ignore")

    binarizer = ColumnTransformer([
        (
            'binarizer',
            internal_binarizer,
            columns_to_binarize
        )
    ])

    one_hot_binarized = Pipeline([
        ("binarizer", binarizer),
        ("one_hot_encoder", internal_encoder_binarizer)
    ])

    # Scaler

    internal_scaler = RobustScaler()
    columns_to_scale = ["imdb_rating, imdb_votes, box_office"]

    scaler = ColumnTransformer([
        ("scaler", internal_scaler, columns_to_scale)
    ])

    # Full pipeline

    feature_engineering_pipeline  = Pipeline([
        (
            "features",
            FeatureUnion([
                ('binaries', one_hot_binarized),
                ('scaled', scaler)
            ])
        )
    ])

    # Machine learning model
    model = RandomForestClassifier(n_estimators=100, warm_start=True)

    # Full pipeline
    final_pipeline = Pipeline([
        ("feature_engineering", feature_engineering_pipeline),
        ("model", model)
    ])

    return final_pipeline


def model_training_validation(final_pipeline: Pipeline, train_x: array,
                              train_y: array, validate_x: array,
                              validate_y: array) -> (Pipeline, dict):
    """ Train and validate model.

    Args:
        final_pipeline (Pipeline): The feature engineering and training
            pipeline.
        train_x (array): The training data.
        train_y (array): The training labels.
        validate_x (array): The validation data.
        validate_y (array): The validation labels.

    Returns:
        final_pipeline (Pipeline): The final pipeline.
        metrics (dict): A dictionary with the model's metrics.
    """

    final_pipeline.fit(train_x, train_y)

    train_pred_y = final_pipeline.predict(train_x)
    validate_pred_y = final_pipeline.predict(validate_x)

    train_accuracy = accuracy_score(train_pred_y, train_y)
    train_recall = recall_score(train_pred_y, train_y)
    validate_accuracy = accuracy_score(validate_pred_y, validate_y)
    validate_recall = recall_score(validate_pred_y, validate_y)

    print('Train accuracy', train_accuracy)
    print('Train recall', train_recall)
    print('Validate accuracy', validate_accuracy)
    print('Validate recall', validate_recall)

    metrics = {
        'train_accuracy': train_accuracy,
        'train_recall': train_recall,
        'validate_accuracy': validate_accuracy,
        'validate_recall': validate_recall,
    }

    return final_pipeline, metrics


def full_training_run() -> Pipeline:
    """ Train the model and return the pipeline.

    Returns:
        Pipeline: The training pipeline.
    """
    training_pipeline = build_pipeline()

