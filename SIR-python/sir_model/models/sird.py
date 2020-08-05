import torch
from torch import nn
from torch.nn import functional as F


class SirdModel(nn.Module):
    def __init__(self, beta=0.02, gamma=0.0093, sigma=0.0006):
        super(SirdModel, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(self, s, i, r, d, n):
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
        new_d = d + self.sigma * i
        new_i = n - new_s - new_r - new_d

        return new_s, new_i, new_r, new_d

    def __repr__(self):
        return f"SirdModel(beta={self.beta}, gamma={self.gamma}, sigma={self.sigma})"
