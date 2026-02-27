import torch
import numpy as np
import random
import os
import sys
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from huggingface_hub import login


def hugging_face_authentication() -> None:
    """Authenticates with Hugging Face using environment variables."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("CRITICAL: HF_TOKEN not found in .env file.")

    login(token=hf_token)


def seed_everything(seed: int) -> None:

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def temperature_annealing(
    epoch: int, temperature_warmup_length: int, temperature_start: float, temperature_end: float) -> float:
    decay = (temperature_start - temperature_end) * (epoch / temperature_warmup_length)
    temperature = max(temperature_end, temperature_start - decay)
    return temperature


def pretty_print_logits_and_weights(logits: torch.Tensor, weights: torch.Tensor) -> None:

    logits_average = logits.mean(dim=(0, 1)).detach().cpu().float().numpy()
    print(f"Average logits for this batch:\n{logits_average}")

    logit_min = logits_average.min()
    logit_max = logits_average.max()
    logit_spread = logit_max - logit_min
    print(f"Logit range: [{logit_min:7.2f}, {logit_max:7.2f}] | Spread: {logit_spread:7.2f}")

    weights_average = weights.mean(dim=(0, 1)).detach().cpu().float().numpy()
    print(f"Average weights for this batch:\n{weights_average}")


def pretty_print_epoch(
    split: str,
    epoch: int,
    nb_epochs: int,
    loss: float,
    epoch_growth: float,
    nb_trading_days_per_year: int
) -> None:

    annualized_returns = 100 * (epoch_growth ** nb_trading_days_per_year -1)

    print("\n[{}] Epoch {}/{}] | Loss: {:.6f} | Portfolio: {:.4f} | Annualized returns: {:.2f}%".format(
        split, epoch, nb_epochs, loss, epoch_growth, annualized_returns))


def pretty_print_datasets(train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:

    nb_samples_train = len(train_dataloader.dataset)
    nb_batches_train = len(train_dataloader)

    nb_samples_test = len(test_dataloader.dataset)
    nb_batches_test = len(test_dataloader)

    print("-"*64)
    print(f"[Train split] {nb_samples_train} samples, {nb_batches_train} batches")
    print(f"[Test split] {nb_samples_test} samples, {nb_batches_test} batches")
    print("-"*64)
