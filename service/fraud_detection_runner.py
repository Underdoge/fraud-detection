""" Create our own bentoml runner for our fraud detection model. """

import bentoml
import pandas as pd


class FraudDetectionModelRunner(bentoml.Runnable):
    """ Define our runner's class properties and is_fraud method. """
    SUPPORTED_RESOURCES = ("cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, model: bentoml.Model) -> None:
        self.classifier = bentoml.sklearn.load_model(model)

    @bentoml.Runnable.method()
    def is_fraud(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """ Get predicted values for input dataset. """
        result = input_data
        predict_probas = self.classifier.predict_proba(input_data)
        negative_proba = predict_probas[:, 1]
        positive_proba = predict_probas[:, 0]
        result["is_fraud_proba"] = negative_proba
        result["is_legit_proba"] = positive_proba
        return result
