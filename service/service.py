""" Use model to generate predictions through an API. """

import bentoml
from bentoml.io import PandasDataFrame
from fraud_detection_runner import FraudDetectionModelRunner

MODEL_TAG = "fraud-detection-model"

fraud_detection_model = bentoml.sklearn.get(MODEL_TAG)
fraud_detection_model_runner = bentoml.Runner(
    FraudDetectionModelRunner,
    models=[fraud_detection_model],
    runnable_init_params={
        "model":fraud_detection_model
    }
)

fraud_detection_service = bentoml.Service("fraud-detection-service",
                                          runners=[fraud_detection_model_runner])


@fraud_detection_service.api(input=PandasDataFrame(), output=PandasDataFrame())

def predict(input_df):
    """ Function to predict on new values that the API receives. """

    return fraud_detection_model_runner.is_fraud.run(input_df)
