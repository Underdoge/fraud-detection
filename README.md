# Proyecto Final para el Bootcamp de Machine Learning de Codigo Facilito 2023


# Installation
Open up a Terminal (macOS/Linux) or PowerShell (Windows) and enter the following commands:
### Cloning the repository
```sh
git clone https://github.com/underdoge/proyecto-final-bcml

cd proyect-final-bcml
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
pip install -r requirements_macos_linux.txt
```
### How to run metaflow pipeline
```sh
python3 fraud_detection_flow.py run --source-file data/credit_card_transactions-ibm_v2.csv
```