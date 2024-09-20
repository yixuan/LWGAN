import torch
import torch.nn as nn

from models.blocks import MLP, CondMLP

# Encoder for toy data sets
# input_dim -> z_dim
class Q_Toy(nn.Module):
    def __init__(self, z_dim, input_dim=2, struct_dim=128):
        super(Q_Toy, self).__init__()

        hidden_dim = [int(i * struct_dim) for i in [8, 4, 2, 1]]
        self.main = MLP(input_dim, hidden_dim, z_dim, act=nn.ReLU(inplace=True))

    def forward(self, inputs):
        output = self.main(inputs)
        return output

# Encoder for toy data sets, with rank restriction
# input_dim + rank -> z_dim
# output[:, rank:] = 0
class Qrank_Toy(nn.Module):
    def __init__(self, z_dim, rank_min, input_dim=2, struct_dim=128):
        super(Qrank_Toy, self).__init__()
        # If the minimum rank is set, we do not need to do a full one-hot encoding for rank
        self.rank_offset = rank_min - 1
        self.max_id = z_dim - self.rank_offset
        hidden_dim = [int(i * struct_dim) for i in [8, 4, 2, 1]]
        self.main = CondMLP(input_dim, hidden_dim, z_dim, self.max_id, act=nn.ReLU(inplace=True))

    def forward(self, inputs, rank: int):
        # id: 1, 2, ..., max_id
        id = rank - self.rank_offset
        cond = torch.zeros(inputs.shape[0], self.max_id, device=inputs.device)
        cond[:, id - 1] = 1.0
        output = self.main(inputs, cond)
        output[:, rank:] = 0.0
        return output

# Generator for toy data sets
# z_dim -> out_dim
class G_Toy(nn.Module):
    def __init__(self, z_dim, struct_dim=128, out_dim=2):
        super(G_Toy, self).__init__()
        hidden_dim = [int(i * struct_dim) for i in [1, 2, 4, 8]]
        self.main = MLP(z_dim, hidden_dim, out_dim, act=nn.SiLU(inplace=True))

    def forward(self, noise):
        output = self.main(noise)
        return output

# Discriminator for toy data sets
# input_dim -> 1
class D_Toy(nn.Module):
    def __init__(self, input_dim=2, struct_dim=128):
        super(D_Toy, self).__init__()
        hidden_dim = [int(i * struct_dim) for i in [4, 2, 1]]
        self.main = MLP(input_dim, hidden_dim, 1, act=nn.ReLU(inplace=True))

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)

# Discriminator for toy data sets, with rank information
# input_dim + rank -> 1
class Drank_Toy(nn.Module):
    def __init__(self, z_dim, rank_min, input_dim=2, struct_dim=128):
        super(Drank_Toy, self).__init__()
        # If the minimum rank is set, we do not need to do a full one-hot encoding for rank
        self.rank_offset = rank_min - 1
        self.max_id = z_dim - self.rank_offset
        hidden_dim = [int(i * struct_dim) for i in [4, 2, 1]]
        self.main = CondMLP(input_dim, hidden_dim, 1, self.max_id, act=nn.ReLU(inplace=True))

    def forward(self, inputs, rank: int):
        # id: 1, 2, ..., max_id
        id = rank - self.rank_offset
        cond = torch.zeros(inputs.shape[0], self.max_id, device=inputs.device)
        cond[:, :(id - 1)] = 1.0
        output = self.main(inputs, cond)
        return output.view(-1)
