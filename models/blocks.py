import math
import torch
import torch.nn as nn

# Generate normal random noise
@torch.jit.script
def gen_noise(n: int, p: int, device: torch.device):
    noise = torch.randn(n, p, device=device)
    return noise

# Generate normal random noise with a given rank
# Other values are set to zero
@torch.jit.script
def gen_noise_with_rank(n: int, p: int, rank: int, device: torch.device):
    noise = torch.zeros(n, p, device=device)
    noise[:, :rank] = torch.randn(n, rank, device=device)
    return noise

# Given a matrix of noise, keep the first `rank` columns unchanged,
# and set other values to zero
@torch.jit.script
def mask_noise(z, rank: int):
    masked = z.clone()
    masked[:, rank:] = 0.0
    return masked

# Used to implement the smoothed ReLU activation function
# https://datascience.stackexchange.com/a/102234
@torch.jit.script
def log_cosh(x):
    return x + nn.functional.softplus(-2.0 * x) - math.log(2.0)

# Smoothed ReLU
class SmoothedReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    @torch.jit.export
    def forward(self, x):
        return log_cosh(self.relu(x))

# A simple class for building MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act):
        super(MLP, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim[0]))
        layers.append(act)
        # Hidden layers
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(act)
        # Last layer
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# A simple class for building MLP with extra variables in each layer
class CondMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, cond_dim, act):
        super(CondMLP, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Linear(in_dim + cond_dim, hidden_dim[0]))
        # Hidden layers
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i] + cond_dim, hidden_dim[i + 1]))
        # Last layer
        last_layer = nn.Linear(hidden_dim[-1] + cond_dim, out_dim)
        self.layers = nn.ModuleList(layers)
        self.last_layer = last_layer
        self.act = act

    def forward(self, x, cond):
        for layer in self.layers:
            x = torch.cat((x, cond), dim=-1)
            x = layer(x)
            x = self.act(x)
        x = torch.cat((x, cond), dim=-1)
        x = self.last_layer(x)
        return x
