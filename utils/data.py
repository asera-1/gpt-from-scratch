# utils/data.py
# ======================================
# Dataset utilities for GPT training
# ======================================

import torch
import random
from typing import List, Tuple

class TokenDataset:
    """
    Takes a FLAT list of token IDs and yields (x, y) windows for next-token prediction.
    """
    def __init__(self, token_ids: List[int], block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

def get_batch(dataset: TokenDataset, batch_size: int):
    idxs = random.sample(range(len(dataset)), k=min(batch_size, len(dataset)))
    xs, ys = [], []
    for i in idxs:
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)
