""" Use the trained model to predict if a transaction is fraudulent or not. """
import polars as pl
import streamlit as st


def preprocess_dataset(data_df):
    """ Preprocess dataset for EDA. """
    new_data_df = data_df.with_columns(
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
    )
    return new_data_df


def predict_transaction(model, data_df):
    """ Use our model to predict if the transaction is fraudulent or legitimate
    and return the results in a dataframe.
    """
    st.single_transact = model.predict(data_df), model.predict_proba(
        data_df)[0]


def predict_file(model, data_df):
    """ Use our model to verify multiple transactions from a file. """
    data_df = preprocess_dataset(data_df)
    st.multiple_transacts = pl.DataFrame({"Predicted_Is_Fraud?": model.predict(data_df)})
