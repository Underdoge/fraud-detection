""" Build our training pipeline with metaflow and register model with
mlflow.
"""

from metaflow import FlowSpec, Parameter, step


class FraudDetectionFlow(FlowSpec):
    """ Class that will contain all our pipeline's steps."""
    source_file = Parameter("source-file", help="Source CSV file")
    train_proportion = Parameter("train-proportion", help="Proportion of the \
dataset to use for training", default=0.6)
    test_proportion = Parameter("test-proportion", help="Proportion of\
 the dataset to use for validation and testing", default=0.5)

    @step
    def start(self):
        """ First step. """
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("/final-project/FraudDetection")
        run = mlflow.start_run()
        self.mlflow_run_id = run.info.run_id

        print("Initializing Pipeline.")
        self.next(self.load_data)

    @step
    def load_data(self):
        """ Load data into polars DataFrame and save it. """
        import mlflow
        import polars as pl

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_id=self.mlflow_run_id):

            self.new_data_df = pl.read_csv(self.source_file)
            self.next(self.preprocess_dataset)

    @step
    def preprocess_dataset(self):
        """ Preprocess dataset. """
        import mlflow
        import polars as pl

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_id=self.mlflow_run_id):

            print("Preprocessing Dataset.")
            new_data_df = self.new_data_df.with_columns(
                Hour = pl.col("Time").map_elements(
                    lambda x: x[:2]).cast(pl.Int64, strict=True),
                Minute = pl.col("Time").map_elements(
                    lambda x: x[3:]).cast(pl.Int64, strict=True),
            ).drop("Time").with_columns(
                pl.col("Amount").map_elements(
                    lambda x: x.replace("$", "")).cast(pl.Float64, strict=True)
            ).with_columns(
                pl.col("Merchant State").fill_null("ONLINE")
            ).with_columns(
                pl.col("Zip").cast(pl.String, strict=True).fill_null("ONLINE")
            ).with_columns(
                pl.col("Errors?").fill_null(value="No")
            ).with_columns(
                pl.col("Is Fraud?").map_elements(
                    lambda x: 0 if x == "No" else 1
                )
            )
            self.new_data_df = new_data_df
            self.next(self.split_dataset)

    @step
    def split_dataset(self):
        """ Convert categorical columns and split dataset. """
        import mlflow
        import polars as pl
        from sklearn.model_selection import train_test_split

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_id=self.mlflow_run_id):

            print("Splitting dataset.")
            data_df = self.new_data_df.with_columns(
                pl.col("Use Chip").cast(pl.Categorical).to_physical()
            ).with_columns(
                pl.col("Merchant Name").cast(pl.String).cast(
                    pl.Categorical).to_physical()
            ).with_columns(
                pl.col("Merchant City").cast(pl.Categorical).to_physical()
            ).with_columns(
                pl.col("Merchant State").cast(pl.Categorical).to_physical()
            ).with_columns(
                pl.col("Zip").cast(pl.Categorical).to_physical()
            ).with_columns(
                pl.col("Errors?").cast(pl.Categorical).to_physical()
            )
            is_fraud = data_df.select(pl.col('Is Fraud?'))
            features = data_df.drop('Is Fraud?')

            original_count = len(data_df)
            training_size = int(original_count * self.train_proportion)
            test_size = int(
                (1 - self.train_proportion) * self.test_proportion * training_size)

            train_x, rest_x, train_y, rest_y = train_test_split(features,
                                                                is_fraud,
                                                                train_size=training_size,
                                                                random_state=0)
            validate_x, test_x, validate_y, test_y  = train_test_split(rest_x,
                                                                    rest_y,
                                                                    train_size=test_size,
                                                                    random_state=0)

            mlflow.log_params({
                'dataset_size': original_count,
                'training_set_size': len(train_x),
                'validate_set_size': len(validate_x),
                'test_set_size': len(test_x)
            })

            self.train_x = train_x
            self.train_y = train_y
            self.validate_x = validate_x
            self.validate_y = validate_y
            self.test_x = test_x
            self.test_y = test_y
            self.next(self.model_training)

    @step
    def model_training(self):
        """ Call our feature pipeline build function and train model. """
        import mlflow
        from imblearn.over_sampling import SMOTE  # noqa: E402

        from feature_pipeline import build_pipeline
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_id=self.mlflow_run_id):

            self.training_pipeline = build_pipeline()
            sm = SMOTE(random_state=0)

            train_x_res, train_y_res = sm.fit_resample(
                self.train_x.to_pandas(), self.train_y.to_pandas())

            self.sm = sm
            self.train_x_res = train_x_res
            self.train_y_res = train_y_res
            self.training_pipeline.fit(train_x_res, train_y_res.to_numpy().ravel())
            self.next(self.model_validation)

    @step
    def model_validation(self):
        """ Test our model against the train, validate and test datasets. """
        import mlflow
        from sklearn.metrics import (  # noqa: E402, F811
            accuracy_score,
            recall_score,
        )
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_id=self.mlflow_run_id):

            validate_x_res, validate_y_res = self.sm.fit_resample(
                self.train_x.to_pandas(), self.train_y.to_pandas())
            test_x_res, test_y_res = self.sm.fit_resample(
                self.train_x.to_pandas(), self.train_y.to_pandas())

            train_pred_y = self.training_pipeline.predict(self.train_x_res)
            validate_pred_y = self.training_pipeline.predict(validate_x_res)
            test_pred_y = self.training_pipeline.predict(test_x_res)

            train_accuracy = accuracy_score(train_pred_y,
                                            self.train_y_res.to_numpy().ravel())
            train_recall = recall_score(train_pred_y,
                                        self.train_y_res.to_numpy().ravel())
            validate_accuracy = accuracy_score(validate_pred_y,
                                            validate_y_res.to_numpy().ravel())
            validate_recall = recall_score(validate_pred_y,
                                        validate_y_res.to_numpy().ravel())
            test_accuracy = accuracy_score(test_pred_y,
                                        test_y_res.to_numpy().ravel())
            test_recall = recall_score(test_pred_y, test_y_res.to_numpy().ravel())

            print("Train accuracy:", train_accuracy)
            print("Train recall:", train_recall)
            print("Validate accuracy:", validate_accuracy)
            print("Validate recall:", validate_recall)
            print("Test accuracy:", test_accuracy)
            print("Test recall:", test_recall)

            metrics = {
                'train_accuracy': train_accuracy,
                'train_recall': train_recall,
                'validate_accuracy': validate_accuracy,
                'validate_recall': validate_recall,
                'test_accuracy': test_accuracy,
                'test_recall': test_recall,
            }
            mlflow.log_metrics(metrics)

            self.next(self.register_model)

    @step
    def register_model(self):
        """ Register model into mlflow. """
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.sklearn.log_model(self.training_pipeline,
                                     "fraud-detection-model")
            mlflow.register_model(
                f"runs:/{self.mlflow_run_id}/fraud-detection-model",
                "fraud-detection-model"
            )
            self.next(self.end)

    @step
    def end(self):
        """ Last step. """
        print("End Pipeline.")


if __name__ == "__main__":
    FraudDetectionFlow()
