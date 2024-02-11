""" Use model to generate predictions through an API. """

import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, PandasDataFrame

MODEL_TAG = "fraud-detection-model"

fraud_detection_model_runner = bentoml.sklearn.get(MODEL_TAG).to_runner()

fraud_detection_service = bentoml.Service("fraud-detection-service", runners=[fraud_detection_model_runner])


@fraud_detection_service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(input_df):
    """ Function to predict on new values that the API receives. """
    # Necesitamos convertir `children` a `np.float64` manualmente debido a las peculiaridades de la serialización JSON
    input_df["children"] = input_df["children"].astype(np.float64)

    return fraud_detection_model_runner.run(input_df)