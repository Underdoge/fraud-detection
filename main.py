""" Contains the data preparation and model training, testing and prediction
methods.
"""

import polars as pl

# from ydata_profiling import ProfileReport
from util.model import full_training_run, predict_oscars

if __name__ == "__main__":
    # pipeline = full_training_run()
    # predict_oscars(pipeline)
    data_df = pl.read_csv("data/credit_card_transactions-ibm_v2.csv", )
    print(data_df)
