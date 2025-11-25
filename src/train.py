from __future__ import annotations

from typing import List, Tuple, Optional

import torch
from torch.optim import SGD

from .loss import mf_loss
from .model import MFModel
from .utils import evaluate_rmse
from .dataset import RatingsDataset


IndexTriple = Tuple[int, int, float]


def train_sgd(
    model: MFModel,
    omega_train: List[IndexTriple],
    omega_val: Optional[List[IndexTriple]],
    ratings_matrix: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 4096,
    alpha: float = 1e-2,
    lambda_: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> None:
    """
    Stochastic Gradient Descent training over random minibatches from Ω.
    - Each step:
        compute loss over minibatch (observed entries only),
        backward,
        optimizer step.
    - Uses autograd; gradient structure matches Eqs. (9.6)–(9.10).
    """
    device = device if device is not None else torch.device("cpu")
    model.to(device)
    ratings_matrix = ratings_matrix.to(device)

    optimizer = SGD(model.parameters(), lr=alpha)

    for epoch in range(1, epochs + 1):
        model.train()
        batches = RatingsDataset.sample_minibatches(omega_train, batch_size=batch_size, seed=epoch)
        running_loss = 0.0

        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = mf_loss(model, batch, ratings_matrix, lambda_=lambda_)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / max(1, len(batches))
        msg = f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.6f}"

        if verbose and omega_val is not None and len(omega_val) > 0:
            model.eval()
            with torch.no_grad():
                val_rmse = evaluate_rmse(model, omega_val, ratings_matrix, device=device)
            msg += f" | Val RMSE: {val_rmse:.6f}"

        if verbose:
            print(msg)


