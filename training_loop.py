import torch
import numpy as np
from torch.utils.data import DataLoader

from misc_utils import pretty_print_logits_and_weights


def get_negative_log_wealth(portfolio_growth: torch.Tensor) -> torch.Tensor:
    """
    Compute an objective function that maximizes log returns.

    Args:
        portfolio_growth: Portfolio return per time step, of shape (batch_size, window_length-1).

    Returns:
        Loss value.
    """
    eps = 1e-8
    # Cast to float64 for numerical stability
    growth_64 = portfolio_growth.to(torch.float64)
    loss = -torch.log(growth_64 + 1e-8).mean()
    return loss


def custom_training_loss_function(
    portfolio_growth: torch.Tensor,
    weights_current: torch.Tensor,
    lambda_entropy: float,
    lambda_turnover: float,
    lambda_volatility: float
) -> torch.Tensor:
    """
    Compute a composite loss

    Args:
        portfolio_growth: Portfolio return per time step, of shape (batch_size, window_length-1).
        weights_current: Portfolio allocations, of shape (batch_size, window_length-1, nb_tickers+1).
        lambda_entropy: Weight for the entropy regularization term.
        lambda_turnover: Weight for the turnover (L1-variation) penalty.

    Returns:
        Loss value.
    """

    # Negative log-wealth: maximize log-returns
    negative_log_wealth = get_negative_log_wealth(portfolio_growth)

    # Entropy: Encourage diversity across tickers
    # Maximizing entropy (via subtraction in the final loss) prevents over-concentration
    entropy = torch.distributions.Categorical(probs=weights_current).entropy().mean()

    # Turnover: Tax temporal variations
    # L1-norm of the difference between consecutive weight vectors
    turnover = torch.norm(weights_current[:, 1:, :] - weights_current[:, :-1, :], p=1, dim=-1).mean()

    # Volatility: Penalize variance across the temporal window
    return_variance = torch.var(portfolio_growth.to(torch.float64), dim=1).mean()

    # Total loss
    loss = (
        negative_log_wealth
        - (lambda_entropy * entropy)
        + (lambda_turnover * turnover)
        + (lambda_volatility * return_variance)
    )

    return loss


def apply_feature_jittering(x: torch.Tensor, lambda_noise: float) -> torch.Tensor:
    """
    Add Gaussian noise to force the model to learn the underlying signal rather than memorizing the exact patterns.
    Increase robustness, apply regularization, prevent over-fitting.
    The operation is conducted in FP32 for precision, values are cast back to BF16 afterwards.
    """
    noise = lambda_noise * torch.randn_like(x, dtype=torch.float32)
    x = (x.float() + noise).to(torch.bfloat16)
    return x


def calculate_portfolio_growth(
    weights: torch.Tensor,
    _raw_logits: torch.Tensor,
    log_returns: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the portfolio growth realized by aligning current allocations with future prices.

    Args:
        weights: Portfolio allocations (including cash), of shape (batch_size, window_length, nb_tickers + 1).
        log_returns: Return of all stocks, in log scale, of shape (batch_size, window_length, nb_tickers).

    Returns:
        Growth factor realized at each time step (from t to t+1), of shape (batch_size, window_length-1).
        Current weights used for calculation, of shape (batch_size, window_length - 1, nb_tickers + 1).
    """

    # Extract the "current" weights, from time step 0 until the penultimate time step (T-1 time points)
    weights_current = weights[:, :-1, :]
    logits_current = _raw_logits[:, :-1, :]

    # Extract the "future" log returns, from time step 1 until the end of the temporal window (T-1 time points)
    log_returns_future = log_returns[:, 1:, :]

    # Express the returns back to linear scale
    stock_returns_future = torch.exp(log_returns_future)

    # Add "Cash" return, of constant value 1.0, to the stock returns (this adds one column next to all existing stocks)
    cash_return = torch.ones_like(stock_returns_future[:, :, :1])
    returns_future = torch.cat([stock_returns_future, cash_return], dim=-1)

    # Calculate portfolio growth, by allocating the current weights to the future returns
    portfolio_growth = torch.sum(weights_current.to(torch.float64) * returns_future.to(torch.float64), dim=-1)

    return portfolio_growth, weights_current, logits_current


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    temperature: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    lambda_entropy: float,
    lambda_turnover: float,
    lambda_noise: float,
    lambda_volatility: float,
    max_gradient_norm: float
) -> tuple[float, float]:
    """
    Execute a single training epoch on the training dataset using policy gradient optimization.

    Args:
        model: Decision Transformer model.
        dataloader: Test dataset loader providing standardized and raw log returns.
        device: Hardware device for computation (CPU/CUDA).
        temperature: Scaling factor for softmax probability distribution.
        optimizer: Optimizer (e.g., AdamW) for weight updates.
        lambda_entropy: Coefficient for the entropy regularization (diversification).
        lambda_turnover: Coefficient for the turnover penalty (trading cost reduction).
        lambda_noise: Standard deviation for Gaussian feature jittering.
        max_gradient_norm: Threshold for gradient clipping.

    Returns:
        Average training compositeloss across all batches.
        Geometric mean of portfolio growth per time step.
    """
    model.train()
    epoch_loss = 0
    total_observations = 0 # Track actual samples processed
    nb_batches = len(dataloader)
    cumulative_growth_log = torch.tensor(0.0, device=device, dtype=torch.float64)

    for batch_index, batch in enumerate(dataloader):

        log_returns_standardized, log_returns = batch
        log_returns_standardized = log_returns_standardized.to(device, dtype=torch.bfloat16, non_blocking=True)
        log_returns = log_returns.to(device, dtype=torch.bfloat16, non_blocking=True)

        log_returns_standardized = apply_feature_jittering(x=log_returns_standardized, lambda_noise=lambda_noise)

        optimizer.zero_grad()

        weights, _raw_logits = model(log_returns_standardized, temperature)

        portfolio_growth, weights_current, logits_current = calculate_portfolio_growth(
            weights=weights, _raw_logits=_raw_logits, log_returns=log_returns)

        batch_size, seq_len, _ = weights_current.shape
        total_observations += (batch_size * seq_len)

        # Cumulative logic: Sum of logs is the log of the product
        cumulative_growth_log += torch.log(portfolio_growth + 1e-8).sum()

        loss = custom_training_loss_function(
            portfolio_growth, weights_current, lambda_entropy, lambda_turnover, lambda_volatility)
        loss.backward()
        epoch_loss += loss.item()

        # Gradient clipping for model stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)

        optimizer.step()

        if batch_index == 0:
            pretty_print_logits_and_weights(logits=logits_current, weights=weights_current)

    # Geometric mean of the growth factor per time step
    epoch_growth = torch.exp(cumulative_growth_log / total_observations).item()

    epoch_loss /= nb_batches

    return epoch_loss, epoch_growth


@torch.no_grad()
def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    temperature: float
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a test dataset to compute portfolio performance metrics.

    Args:
        model: Decision Transformer model.
        dataloader: Test dataset loader providing standardized and raw log returns.
        device: Hardware device for computation (CPU/CUDA).
        temperature: Scaling factor for softmax probability distribution.

    Returns:
        Mean negative log wealth across all batches.
        Geometric mean of portfolio growth per time step.
        Array of portfolio weights at each time step, of shape (nb_time_steps, nb_tickers +1).
        Compounded portfolio value over time, starting from 1.0, of shape (nb_time_steps).
    """
    model.eval()
    total_observations = 0 # Track actual samples processed
    epoch_loss = 0
    nb_batches = len(dataloader)
    cumulative_growth_log = torch.tensor(0.0, device=device, dtype=torch.float64)
    weights_history = []
    cumulative_gains_history = []
    running_log_wealth = 0.0

    for batch_index, batch in enumerate(dataloader):

        log_returns_standardized, log_returns = batch
        log_returns_standardized = log_returns_standardized.to(device, dtype=torch.bfloat16, non_blocking=True)
        log_returns = log_returns.to(device, dtype=torch.bfloat16, non_blocking=True)

        weights, _raw_logits = model(log_returns_standardized, temperature)

        portfolio_growth, weights_current, logits_current = calculate_portfolio_growth(
            weights=weights, _raw_logits=_raw_logits, log_returns=log_returns)

        # Retrieve the weights attribution, of shape (batch_size, nb_tickers +1)
        weights_history_batch = weights_current[:, -1, :].cpu().float()

        # Store the weights attribution
        weights_history.append(weights_history_batch)

        # Update cumulative gains step by step
        last_step_returns = portfolio_growth[:, -1].cpu().float()
        for r in last_step_returns:
            running_log_wealth += np.log(r.item() + 1e-8)
            cumulative_gains_history.append(np.exp(running_log_wealth))

        batch_size, seq_len, _ = weights_current.shape
        total_observations += (batch_size * seq_len)

        # Cumulative Logic: Sum of logs is the log of the product
        cumulative_growth_log += torch.log(portfolio_growth + 1e-8).sum()

        loss = get_negative_log_wealth(portfolio_growth)
        epoch_loss += loss.item()

        if batch_index == 0:
            pretty_print_logits_and_weights(logits=logits_current, weights=weights_current)

    # Geometric mean of the growth factor per time step
    epoch_growth = torch.exp(cumulative_growth_log / total_observations).item()

    epoch_loss /= nb_batches

    weights_history = torch.cat(weights_history, dim=0).numpy()

    cumulative_gains_history = np.array(cumulative_gains_history)

    return epoch_loss, epoch_growth, weights_history, cumulative_gains_history
