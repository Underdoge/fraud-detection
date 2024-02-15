# Fraud Detection 🔎
## Final Project for the Código Facilito's 2023 Machine Learning Bootcamp
# Hosted Streamlit App
A hosted version of the model can be found on Streamlit [here](https://fraud-detection-underdoge.streamlit.app).
#
# Installation
Open up a Terminal and enter the following commands:
### Cloning the repository
```sh
git clone https://github.com/underdoge/fraud-detection

cd fraud-detection
```
### Creating the virtual environment
```sh
python -m venv .venv
```
### Activating the virtual environment
```sh
source .venv/bin/activate
```
### Installing on macOS / Linux
```sh
pip install -r requirements.txt
```
### Launching the mlflow server
```sh
mlflow server
```
### Running the metaflow pipeline (requires mlflow server running)
```sh
python fraud_detection_flow.py run --source-file data/credit_card_transactions-ibm_v2.csv
```
### Import mlflow model into bentoml
```sh
python import_mlflow_model.py
```
### Verify model was successfully imported in bentoml
```sh
bentoml models list
```
### Launch bentoml service
```sh
cd service
bentoml serve service.py --reload
``` 
### Build Bento
```sh
bentoml build
```
### List created Bentos
```sh
bentoml list
```
### Create Docker image from Bento (requires Docker)
```sh
bentoml containerize <model tag from previous step e.g. fraud-detection-service:worn7ggjg2q63yqs>
```
### Run model from Docker image (requires Docker)
```sh
docker run -p 3000:3000 <model tag from previous step e.g. fraud-detection-service:worn7ggjg2q63yqs>
```
# Requirements
- Python 3.11.6 or greater
- Git (to clone the repo)
- Docker (to create Docker image)
#
# Dataset Sources
- [Credit Card Transactions Kaggle Dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)