import torch
from torch import nn
from torch.nn import functional as F


class SirModel(nn.Module):
    def __init__(self):
        super(SirModel, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.02))
        self.gamma = nn.Parameter(torch.tensor(0.001))

    def forward(self, s, i, r, n):
        """
        Args:
            s (int)
            i (int)
            r (int)
            n (int)
        Return:
            new_s, new_i, new_r
        """
        ds = -s * i / n
        new_s = s + self.beta * ds

        new_r = r + self.gamma * i
        new_i = n - new_s - new_r

        return new_s, new_i, new_r

    def __repr__(self):
        return f"SirModel(beta={self.beta}, gamma={self.gamma})"
