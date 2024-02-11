""" Download the model from mlflow registry to use it in bentoml. """

import bentoml
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

model_name = "fraud-detection-model"
model_version = 1
model_mlflow_uri = f"models:/{model_name}/{model_version}"

sklearn_model = mlflow.sklearn.load_model(model_uri=model_mlflow_uri)

bento_model = bentoml.sklearn.save_model(name=f"{model_name}:{model_version}",
                                         model=sklearn_model)
