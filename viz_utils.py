import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_stocks_performance(df: pd.DataFrame, split_index: int) -> None:

    save_path = "images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    label_names = df.columns.tolist()
    nb_tickers = len(df.columns)

    cumulative_returns = df / df.iloc[0]

    cmap = plt.get_cmap("gist_rainbow")
    ticker_colors = [cmap(i) for i in np.linspace(0, 1, nb_tickers)]
    color_map = {ticker: ticker_colors[i] for i, ticker in enumerate(label_names)}

    plt.figure(figsize=(14, 7))

    for ticker in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker, alpha=0.7, color=color_map[ticker])

    benchmark = cumulative_returns.mean(axis=1)
    plt.plot(cumulative_returns.index, benchmark, color="black", linewidth=3, label="Equal weight index")

    plt.axvline(x=df.index[split_index], color="black", linestyle="--", linewidth=3, label="Train/Test Split")

    plt.title("Stocks performance (Normalized)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Growth of 1 € investment", fontsize=12)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "stocks_performance.png"), dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_weight_attribution(weights_history, cumulative_gains_history, label_names, epoch, price_df, dates) -> None:

    save_path = "images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _, nb_tickers_plus_cash = weights_history.shape
    nb_tickers = nb_tickers_plus_cash - 1

    # Ensure prices are aligned with the test window dates
    # Assuming price_df contains all tickers in label_names (excluding CASH)
    test_prices = price_df.loc[dates]

    cumulative_returns = test_prices / test_prices.iloc[0]

    cmap = plt.get_cmap("gist_rainbow")
    ticker_colors = [cmap(i) for i in np.linspace(0, 1, nb_tickers)]

    color_map = {ticker: ticker_colors[i] for i, ticker in enumerate(label_names) if ticker != "CASH"}
    color_map["CASH"] = "black"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top subplot
    for ticker in cumulative_returns.columns:
        ax1.plot(dates, cumulative_returns[ticker],color=color_map.get(ticker, "gray"), alpha=0.7, label=ticker)

    benchmark = cumulative_returns.mean(axis=1)
    ax1.plot(cumulative_returns.index, benchmark, color="black", linewidth=3, label="Equal weight index")

    ax1.plot(
        cumulative_returns.index, cumulative_gains_history, color="crimson", linewidth=4, label="Model portfolio (cumulative)")

    ax1.set_title(f"Market performance (Test set, Epoch {epoch})", fontsize=14)
    ax1.set_ylabel("Normalized growth", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Bottom subplot
    df_weights = pd.DataFrame(weights_history, index=dates, columns=label_names)

    for column in df_weights.columns:
        if column == "CASH":
            ax2.plot(dates, df_weights[column], label=column, linewidth=2.5, color=color_map[column], linestyle="--")
        else:
            ax2.plot(dates, df_weights[column], color=color_map[column], alpha=0.8, linewidth=1.5)

    # Equal weight reference line
    ax2.axhline(y=1.0/nb_tickers_plus_cash, color="black", linestyle=":", alpha=0.8, label="Equal weight assignment")

    ax2.set_title(f"Model weight attribution (Test set, Epoch {epoch})", fontsize=14)
    ax2.set_ylabel("Portfolio weight", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    handles = handles_1 + handles_2
    labels = labels_1 + labels_2
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    plt.savefig(os.path.join(save_path, f"weight_attribution_for_epoch_{epoch}.png"), dpi=300, bbox_inches="tight")
    plt.close("all")

