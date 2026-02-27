import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from misc_utils import hugging_face_authentication, seed_everything, pretty_print_epoch, temperature_annealing
from portfolio_dataset import PortfolioDataset
from transformer_model import TransformerModel
from portfolio_dataset import get_dataloaders
from training_loop import train_step, test_step
from viz_utils import plot_weight_attribution


if __name__ == "__main__":

    # ----------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------

    # Model
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    # Dataset: stock prices
    window_length = 50  # Number of trading days (5 days per week)
    train_test_split = 0.75
    period = "3y"
    interval = "1d"


    # Overall training
    nb_epochs = 20
    batch_size = 36

    # Label smoothing (higher temperature: more diversified; lower temperature: sparser)
    temperature_start = 2.0  # Temperature at the start of the warm-up period
    temperature_end = 0.25  # Temperature at the end of the warm-up period, and onward
    temperature_warmup_length = 20  # Number of epochs for temperature warm-up

    # Gradient clipping for stability during the training phase
    max_gradient_norm = 0.5

    # Logits clamping: Cap the strong logits to maintain the weak logits alive
    logits_clamp_value = 10.0

    # Training phase
    lambda_noise = 0.03  # Feature jittering (Gaussian noise addition)
    lambda_entropy = 0.005  # Diversity: higher values encourage more spread across tickers
    lambda_turnover = 0.035  # Trading fees: higher values maintain temporal consistency
    lambda_volatility = 0.35  # Penalize variance over temporal windows

    # Optimizer
    weight_decay_head = 0.8  # Regularization
    weight_decay_other = 0.01  # Regularization
    learning_rate_head = 1.2e-4  # The head must bridge the gap between LLM embeddings and asset weights
    learning_rate_other = 7e-5  # LoRA parameters are fine-tuned slowly to avoid over-fitting

    # Misc
    seed = 421
    nb_trading_days_per_year = 252
    # ----------------------------------------------------------------

    hugging_face_authentication()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    seed_everything(seed=seed)

    train_dataloader, test_dataloader, nb_tickers, label_names, dates_test, stock_prices = get_dataloaders(
        window_length=window_length,
        train_test_split=train_test_split,
        period=period,
        interval=interval,
        batch_size=batch_size)

    model = TransformerModel(
        model_id=model_id,
        nb_tickers=nb_tickers,
        logits_clamp_value=logits_clamp_value)

    model.to(device)
    model.to(torch.bfloat16)

    # Setup Optimizer: the head is separated from the rest of the model (LoRA + Encoder)
    head_params = [p for n, p in model.named_parameters() if "action_logits" in n]
    other_params = [p for n, p in model.named_parameters() if "action_logits" not in n]
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'weight_decay': weight_decay_head, 'lr': learning_rate_head},
        {'params': other_params, 'weight_decay': weight_decay_other, 'lr': learning_rate_other}
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=nb_epochs)

    for epoch in range(nb_epochs):

        temperature = temperature_annealing(
            epoch=epoch,
            temperature_warmup_length=temperature_warmup_length,
            temperature_start=temperature_start,
            temperature_end=temperature_end)

        # Train
        loss_train, epoch_growth_train = train_step(
            model=model,
            dataloader=train_dataloader,
            device=device,
            temperature=temperature,
            optimizer=optimizer,
            lambda_entropy=lambda_entropy,
            lambda_turnover=lambda_turnover,
            lambda_noise=lambda_noise,
            lambda_volatility=lambda_volatility,
            max_gradient_norm=max_gradient_norm
        )

        pretty_print_epoch(
            split="train",
            epoch=epoch,
            nb_epochs=nb_epochs,
            loss=loss_train,
            epoch_growth=epoch_growth_train,
            nb_trading_days_per_year=nb_trading_days_per_year
        )

        # Test
        loss_test, epoch_growth_test, test_weights_history, cumulative_gains_history = test_step(
            model=model,
            dataloader=test_dataloader,
            device=device,
            temperature=temperature
        )

        pretty_print_epoch(
            split="test",
            epoch=epoch,
            nb_epochs=nb_epochs,
            loss=loss_test,
            epoch_growth=epoch_growth_test,
            nb_trading_days_per_year=nb_trading_days_per_year
        )

        plot_weight_attribution(
            weights_history=test_weights_history,
            cumulative_gains_history=cumulative_gains_history,
            label_names=label_names,
            epoch=epoch,
            price_df=stock_prices,
            dates=dates_test
        )

        scheduler.step()
