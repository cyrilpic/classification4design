import math
import torch
from torch import Tensor


def ackley(X: Tensor) -> Tensor:
    a, b, c = 20, 0.2, 2 * math.pi
    part1 = -a * torch.exp(-b / math.sqrt(X.shape[-1]) * torch.norm(X, dim=-1))
    part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
    return part1 + part2 + a + math.e


def four_islands(X: Tensor) -> Tensor:
    g1 = 50 * (-0.3 * X[:, :1] + 0.7 * X[:, 1:2]) - 25.0
    g2 = (X[:, :1] - 0.2) ** 2 + (X[:, 1:2] - 0.7) ** 2 - 0.25
    g3 = torch.exp((X[:, :1] * 0.6 + X[:, 1:2] * 0.4) ** 2 + 0.3) - 1.6
    g4 = 0.5 * (X[:, :1]) ** 2 + (X[:, 1:2]) ** 2 - 0.21
    g5 = ackley(X - torch.tensor([0.0, 0.3]).to(X)) - 2.0
    g6 = -70 * (X[:, :1] - 0.12) ** 2 + 0.22
    g7 = -70 * (X[:, 1:2] - 0.33) ** 2 + 0.3
    return torch.concat([g1, g2, g3, g4, g5[:, None], g6, g7], axis=1)
