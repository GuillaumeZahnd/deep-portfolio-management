# Portfolio Management

Transformer-based reinforcement learning for context-aware portfolio policy optimization.

<img width="4163" height="2068" alt="stocks_performance" src="https://github.com/user-attachments/assets/15c261ce-9d8a-4bb5-8a41-88147dc3a883" />

The purpose of this script is to learn patterns in financial time series (21 stocks across 7 different sectors) and to predict how to optimally distribute the investment weights to optimize return. The script is evaluated using a test set (previously unseen data) and demonstrates a substantial improvement against the equal weights index.

<img width="4192" height="2964" alt="weight_attribution_for_epoch_20" src="https://github.com/user-attachments/assets/0c20b0d1-f4d1-487c-ab3d-1cee90e53c92" />

## Setup

### Environment initialization

```sh
python -m pip install --upgrade setuptools pip
pipenv install -d --python 3.12
```

### Environment variable

Edit the file `env_template` to add your Hugging Face token, and rename the file to `.env`.

### Environment activation

```sh
pipenv shell
``` 

## Running the main script

```sh
python main.py
```
