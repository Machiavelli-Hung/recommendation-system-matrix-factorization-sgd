from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .model import MFModel


IndexTriple = Tuple[int, int, float]


def evaluate_rmse(model: MFModel, omega: List[IndexTriple], M: torch.Tensor, device: torch.device | None = None) -> float:
    # Ensure M and indices live on the same device as the model (or provided device)
    # RMSE (Root Mean Squared Error) is the square root of MSE (Mean Squared Error),
    # computed only over the known entries (Ω) of the matrix.
    #
    # Formula:
    #    RMSE = sqrt( (1 / |Ω|) * Σ_{(i,j) ∈ Ω} (M_ij - M_hat_ij)^2 )
    #
    # Where:
    #    M_ij      : true value at position (i,j)
    #    M_hat_ij  : predicted value at position (i,j)
    #    |Ω|       : number of known entries in matrix M
    #
    # Meaning:
    #    - Smaller RMSE → more accurate predictions
    #    - RMSE has the same units as the original data
    device = device if device is not None else model.A.device
    if len(omega) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        user_idx = torch.tensor([i for (i, _, _) in omega], dtype=torch.long, device=device)
        item_idx = torch.tensor([j for (_, j, _) in omega], dtype=torch.long, device=device)
        M_dev = M.to(device)
        targets = M_dev[user_idx, item_idx]
        preds = model(user_idx, item_idx)
        mse = torch.mean((targets - preds) ** 2).item()
    return mse ** 0.5


def save_checkpoint(
    path: str,
    model: MFModel,
    user_id_to_index: Dict[str, int] | None = None,
    item_id_to_index: Dict[str, int] | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "num_users": model.num_users,
        "num_items": model.num_items,
        "k": model.k,
        "user_id_to_index": user_id_to_index,
        "item_id_to_index": item_id_to_index,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: MFModel) -> None:
    payload = torch.load(path, map_location=model.A.device)
    model.load_state_dict(payload["state_dict"])


