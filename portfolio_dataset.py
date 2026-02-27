import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from viz_utils import plot_stocks_performance
from misc_utils import pretty_print_datasets


def get_stock_prices(period: str, interval: str) -> pd.DataFrame:

    tickers = [
        # Information Technology (Apple, Microsoft, Nvidia)
        "AAPL", "MSFT", "NVDA",
        # Financials (JPMorgan Chase, Goldman Sachs, Visa)
        "JPM", "GS", "V",
        # Healthcare (Johnson & Johnson, UnitedHealth Group, Pfizer)
        "JNJ", "UNH", "PFE",
        # Consumer Discretionary (Amazon, Tesla, McDonald's)
        "AMZN", "TSLA", "MCD",
        # Energy (ExxonMobil, Chevron, Schlumberger)
        "XOM", "CVX", "SLB",
        # Consumer Staples (Walmart, Coca-Cola, Procter & Gamble)
        "WMT", "KO", "PG",
        # Industrials (Boeing, Caterpillar, GE Vernova)
        "BA", "CAT", "GE"
    ]

    data = yf.download(tickers, period=period, interval=interval)
    stock_prices = data["Close"]
    stock_prices = stock_prices.ffill().dropna()

    return stock_prices


class PortfolioDataset(Dataset):
    """Dataset for sequence-based portfolio management training."""
    def __init__(self, log_returns: pd.DataFrame, window_length: int, data_mean: float, data_std: float) -> None:

        self.log_returns = log_returns.values

        if data_mean is None or data_std is None:
            # For the training set: dataset is known, we calculate the mean and std
            self.data_mean = self.log_returns.mean(axis=0)
            self.data_std = self.log_returns.std(axis=0) + 1e-6
        else:
            # For the test set: dataset is a priori unknown, we use the mean and std from the training set instead
            self.data_mean = data_mean
            self.data_std = data_std

        # Standardized returns for model input stability
        self.log_returns_standardized = (self.log_returns - self.data_mean) / self.data_std

        self.window_length = window_length
        self.nb_tickers = self.log_returns.shape[1]


    def __len__(self) -> int:
        return self.log_returns.shape[0] - self.window_length


    def __getitem__(self, t: int):
        window_log_returns_standardized = self.log_returns_standardized[t:t+self.window_length]
        window_log_returns = self.log_returns[t:t+self.window_length]

        window_log_returns_standardized = torch.tensor(window_log_returns_standardized, dtype=torch.float32)
        window_log_returns = torch.tensor(window_log_returns, dtype=torch.float32)

        return window_log_returns_standardized, window_log_returns


def get_dataloaders(
    window_length: int,
    train_test_split: float,
    period: str,
    interval: str,
    batch_size: int
) -> tuple[DataLoader, DataLoader, int]:

    stock_prices = get_stock_prices(period=period, interval=interval)

    # Calculate log returns: ln(P_t / P_{t-1})
    log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()

    split_index = int(len(log_returns) * train_test_split)

    log_returns_train = log_returns.iloc[:split_index]
    log_returns_test = log_returns.iloc[split_index:]

    train_dataset = PortfolioDataset(
        log_returns=log_returns_train,
        window_length=window_length,
        data_mean=None,
        data_std=None
    )

    test_dataset = PortfolioDataset(
        log_returns=log_returns_test,
        window_length=window_length,
        data_mean=train_dataset.data_mean,
        data_std=train_dataset.data_std
    )

    nb_tickers = train_dataset.nb_tickers

    label_names = stock_prices.columns.tolist()
    label_names.append("CASH")
    dates_test = log_returns_test.index[window_length:]


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    plot_stocks_performance(df=stock_prices, split_index=split_index)
    pretty_print_datasets(train_dataloader, test_dataloader)

    return train_dataloader, test_dataloader, nb_tickers, label_names, dates_test, stock_prices
