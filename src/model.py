from __future__ import annotations

import torch
from torch import nn


class MFModel(nn.Module):
    """
    Matrix Factorization model approximating M ≈ A B^T
    - A: (num_users, k)
    - B: (num_items, k)

    Forward returns dot products for provided (user_indices, item_indices) pairs:
      pred_ij = A[i] · B[j]
    """

    def __init__(self, num_users: int, num_items: int, k: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.k = k

        device = device if device is not None else torch.device("cpu")

        # Initialize parameters with small random values
        A = torch.randn(num_users, k, dtype=torch.float32, device=device) * 0.01
        B = torch.randn(num_items, k, dtype=torch.float32, device=device) * 0.01

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

    def forward(self, user_indices: torch.LongTensor, item_indices: torch.LongTensor) -> torch.Tensor:
        # Compute dot products A[user] · B[item]
        a = self.A[user_indices]  # (batch, k)
        b = self.B[item_indices]  # (batch, k)
        return torch.sum(a * b, dim=1)  # (batch,)


