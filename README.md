# Portfolio Management

Transformer-based reinforcement learning for context-aware portfolio policy optimization.

The purpose of this script is to learn patterns in financial time series (21 stocks across 7 different sectors) and to predict how to optimally distribute the investment weights to optimize return. The script is evaluated using a test set (previously unseen data) and demonstrates a substantial improvement against the equal weights index.

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
