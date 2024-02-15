"""Streamlit app code.

This Streamlit app will showcase the use of the 'inference_pipeline.joblib'
model under the 'model' folder.
"""
import polars as pl
import streamlit as st
from joblib import load  # noqa: E402

from util.is_fraud import predict_file, predict_transaction

# "st.session_state object:", st.session_state

model = load('model/inference_pipeline.joblib')

st.markdown(
    """
# Fraud Detection :mag:

Welcome to our fraud detection app!

Below you will find two ways to test our fraud detection model.

## Manual transaction check

Here you can enter the information of a transaction to see if our model
would flag it as fraudulent or not, and the probability of each case.

### Transaction Info:
"""
)

countries_df = pl.read_csv("app_data/countries.csv")
states_df = pl.read_csv("app_data/states.csv")
cities_df = pl.read_csv("app_data/cities.csv")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date")
    amount = st.number_input("Amount", value=10.0)
    chip_info = st.radio("Transaction Type",
                     ["Online", "Chip", "Swipe"],
                     captions=["Online Transaction",
                               "Physical card - Card's chip used",
                      "Physical card - Card was swiped"])
with col2:
    time = st.time_input("Time")
    mcc = st.number_input("MCC - [Merchant Category Code](https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf)",
                      placeholder="Merchant Category Code...",
                      value=5411, min_value=1, max_value=9_999,
                      format="%i")
    errors = st.multiselect("Transaction Errors",
                        ['Bad Card Number', 'Bad CVV',
                         'Bad Expiration', 'Insufficient Balance',
                         'Bad Zipcode', 'Technical Glitch', 'Bad PIN'],
                         placeholder="Select one or more errors, if any...")

city = None
if chip_info == "Online":
    country = "ONLINE"
    state = "ONLINE"
    zip = "ONLINE"
else:
    countries = countries_df.select(
        pl.col("name")
    ).to_series().to_list()
    country = st.selectbox("Country", countries, index=0,
                        placeholder="Country where the transaction was made.")
    country_id = countries_df.filter(
        pl.col("name") == country
    ).select(
        pl.col("id")
    ).to_numpy()[0][0]
    if country_id == 233:  #  233 = United States
        states = states_df.filter(
            pl.col("country_id") == country_id
        ).select(
            pl.col("state_code")
        ).to_series().to_list()
        state_code = st.selectbox("State", states, index=0,
                        placeholder="State where the transaction was made.")
        state_id = states_df.filter(
            pl.col("country_id") == 233
        ).filter(pl.col("state_code") == state_code).select(
            pl.col("id")
        ).to_numpy()[0][0]
        cities = cities_df.filter(
            pl.col("state_id") == state_id
        ).select(
            pl.col("name")
        ).to_series().to_list()
        city = st.selectbox("Cities", cities, index=0,
                            placeholder="City where the transaction was made.")
        zip = st.number_input("Zip Code", value=91750, format="%d",
                              min_value=1, max_value=99_999)
        state = state_code
    else:
        states = states_df.filter(
            pl.col("country_id") == country_id
        ).select(
            pl.col("name")
        ).to_series().to_list()
        if len(states) > 0:
            state = st.selectbox("State", states,
                        placeholder="State where the transaction was made.")
            state_id = states_df.filter(
                pl.col("name") == state
            ).select(
                pl.col("id")
            ).to_numpy()[0][0]
            cities = cities_df.filter(
                pl.col("state_id") == state_id
            ).select(
                pl.col("name")
            ).to_series().to_list()
            if len(cities) > 0:
                city = st.selectbox("Cities", cities, index=0,
                            placeholder="City where the transaction was made.")
            else:
                city = state
        else:
            state = country
        zip = st.number_input("Zip Code", value=91750, format="%d",
                              min_value=1, max_value=99_999)

hour = time.hour
minute = time.minute

if errors is not None:
    for idx, error in enumerate(errors):
        if idx < len(errors)-1:
            errors[idx] = error + ","
    errors = "".join(errors)
card = 1

data_df = pl.DataFrame({"Card": 1,
                        "Amount": amount,
                        "Merchant Name": 4645744106416199425,
                        "Merchant State": state,
                        "Merchant City": city,
                        "Zip": zip,
                        "MCC": mcc,
                        "Errors?": errors,
                        "Hour": hour,
                        "Minute": minute,
                        "Use Chip": chip_info + " Transaction"})

st.markdown(
    """
Press the "Verify Transaction" button to check if the transaction is\
 legitimate:
"""
)

st.button("Verify Transaction", on_click=predict_transaction(model, data_df))
if st.single_transact is not None:
    print(st.single_transact)
    if st.single_transact[0] == 0:
        st.markdown(f":white_check_mark: The transaction is legitimate,\
            with a {st.single_transact[1][0]} probability!")
    else:
        st.markdown(f":x: The transaction is fraulent,\
                    with a {st.single_transact[1][1]} probability!")

st.markdown(
    """
## Batch transaction check

Here you can upload a CSV file with multiple transactions.
Click [here](https://raw.githubusercontent.com/Underdoge/\
fraud-detection/main/app_data/test_data.csv) to download a sample
template file.
"""
)

uploaded_file = st.file_uploader("Upload a CSV file")
if uploaded_file is not None:
    data_df = pl.read_csv(uploaded_file)
    predict_file(model, data_df)
    final_data_df = pl.concat([data_df, st.multiple_transacts],
                              how="horizontal")
    st.markdown("Here's the original CSV file with a new\
 'Predicted_Is_Fraud?' column with the prediction for each transaction:")
    st.dataframe(final_data_df)
