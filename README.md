# Proyecto Final para el Bootcamp de Machine Learning de Codigo Facilito 2023


# Installation
Open up a Terminal (macOS/Linux) or PowerShell (Windows) and enter the following commands:
### Cloning the repository
```sh
git clone https://github.com/underdoge/proyecto-final-bcml

cd proyecto-final-bcml
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