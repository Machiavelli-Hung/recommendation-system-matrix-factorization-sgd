from __future__ import annotations

from typing import List, Tuple

import torch

from .model import MFModel


IndexTriple = Tuple[int, int, float]


def mf_loss(model: MFModel, batch: List[IndexTriple], M: torch.Tensor, lambda_: float = 0.0) -> torch.Tensor:
    """
    Custom MF loss over a minibatch of Ω:
      L(A,B) = (1/|Ω_batch|) * Σ_{(i,j)∈Ω_batch} (M_ij - A_i·B_j)^2 + (λ/2)(||A||_F^2 + ||B||_F^2)

    Notes:
    - The data term matches Eq. (9.5): average squared error over observed entries only.
    - The regularizer yields gradients λ A and λ B (Eqs. (9.6)–(9.10) structure).
    """
    device = model.A.device
    if len(batch) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    user_idx = torch.tensor([i for (i, _, _) in batch], dtype=torch.long, device=device)
    item_idx = torch.tensor([j for (_, j, _) in batch], dtype=torch.long, device=device)
    targets = M[user_idx, item_idx]  # (batch,)

    preds = model(user_idx, item_idx)  # (batch,)
    data_term = torch.mean((targets - preds) ** 2)

    if lambda_ > 0.0:
        reg_term = 0.5 * lambda_ * (torch.sum(model.A ** 2) + torch.sum(model.B ** 2))
    else:
        reg_term = torch.tensor(0.0, dtype=torch.float32, device=device)

    return data_term + reg_term


