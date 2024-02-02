""" Methods to build the model's feature engineering and training pipeline. """

import polars as pl
from numpy import array
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from util.omdb_api import process_movies


def build_pipeline() -> Pipeline:
    """ Creates feature engineering and training pipeline.

    Returns:
        final_pipeline (Pipeline): The final pipeline.
    """

    # Scaler
    internal_scaler = RobustScaler()
    columns_to_scale = ["imdb_rating", "imdb_votes"]

    scaler = ColumnTransformer([
        ("scaler", internal_scaler, columns_to_scale)
    ])

    # Full pipeline
    feature_engineering_pipeline  = Pipeline([
        (
            "features",
            FeatureUnion([
                ('scaled', scaler)
            ])
        )
    ])

    # Machine learning model
    model = SVC(probability=True)

    # Full pipeline
    final_pipeline = Pipeline([
        ("feature_engineering", feature_engineering_pipeline),
        ("model", model)
    ])

    return final_pipeline, model


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

    metrics = {
        'train_accuracy': train_accuracy,
        'train_recall': train_recall,
        'validate_accuracy': validate_accuracy,
        'validate_recall': validate_recall,
    }

    return final_pipeline, metrics


def full_training_run() -> Pipeline:
    """ Train the model and return the pipeline.

    We will use each year's Oscars event to train the model separately, and use
    the next year's event for validation until we reach the last event. We will
    start with 46th Oscars event since that's where we have most of the data
    available. We will skip years where the winner was removed due to zero
    values.

    Returns:
        Pipeline: The training pipeline.
    """
    training_pipeline, model = build_pipeline()

    previous_years_df, new_nominees_df  = process_movies(
        "data/nominees_with_metadata_by_year.csv")

    start_oscars_event = 73
    final_oscars_event = previous_years_df.select(
        pl.col("Oscars_ID").last()).to_numpy()[0][0]

    clean_previous_years_df = previous_years_df.filter(
        pl.col("Oscars_ID") == 73
    )

    for oscars_event in range(start_oscars_event+1, final_oscars_event):
        if (len(previous_years_df.filter(
            pl.col("Oscars_ID") == oscars_event).filter(
                pl.col("Won") == 1
            )) > 0):
            clean_previous_years_df.vstack(
                previous_years_df.filter(pl.col("Oscars_ID") == oscars_event),
                in_place=True
            )

    total_events = clean_previous_years_df["Oscars_ID"].unique().to_list()
    training_split = int(.8*len(total_events))
    training_df = clean_previous_years_df.filter(
        pl.col("Oscars_ID") < total_events[training_split]
    )
    print("Training DF:", training_df)
    training_x = training_df.select(
        pl.col("imdb_rating"), pl.col("imdb_votes")
    )
    training_y = training_df.select(
        pl.col("Won")
    )
    validation_df = clean_previous_years_df.filter(
        pl.col("Oscars_ID") >= total_events[training_split]
    )
    print("Validation DF:", validation_df)
    validation_x = validation_df.select(
        pl.col("imdb_rating"), pl.col("imdb_votes")
    )
    validation_y = validation_df.select(
        pl.col("Won")
    )

    training_pipeline, metrics = model_training_validation(
        training_pipeline,
        train_x=training_x,
        train_y=training_y.to_numpy().ravel(),
        validate_x=validation_x,
        validate_y=validation_y.to_numpy().ravel()
    )
    print("Metrics:", metrics)

    return training_pipeline


def predict_oscars(final_pipeline: Pipeline) -> list:
    """ Ok this is a test.

    Args:
        final_pipeline (Pipeline): _description_

    Returns:
        list: _description_
    """
    previous_nominees_df, new_nominees_df  = process_movies(
        "data/nominees_with_metadata_by_year.csv")
    print("Predicting for the following event:")

    # print(new_nominees_df)
    # new_nominees_df = new_nominees_df.select(
    #     pl.col("imdb_rating"),
    #     pl.col("imdb_votes"),
    #     )
    # predict_oscars = final_pipeline.predict(new_nominees_df)
    # predict_proba = final_pipeline.predict_proba(new_nominees_df)

    previous_nominees_df = previous_nominees_df.filter(pl.col("Oscars_ID") == 95)
    print(previous_nominees_df)
    previous_nominees_df = previous_nominees_df.select(
        pl.col("imdb_rating"),
        pl.col("imdb_votes"),
        pl.col("box_office")
        )
    predict_oscars = final_pipeline.predict(previous_nominees_df)
    predict_proba = final_pipeline.predict_proba(previous_nominees_df)

    print("Prediction:", predict_oscars)
    print("Proba:", predict_proba)
