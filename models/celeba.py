import torch
import torch.nn as nn

from models.blocks import SmoothedReLU

# Select activation function
def act_select(act):
    if act == "relu":
        act, act_inp = nn.ReLU(), nn.ReLU(inplace=True)
    elif act == "silu":
        act, act_inp = nn.SiLU(), nn.SiLU(inplace=True)
    elif act == "leakyrelu":
        act, act_inp = nn.LeakyReLU(0.2), nn.LeakyReLU(0.2, inplace=True)
    elif act == "srelu":
        act, act_inp = SmoothedReLU(), SmoothedReLU(inplace=True)
    else:
        raise RuntimeError("unknown activation function")
    return act, act_inp

# Encoder
class Q_CelebA(nn.Module):
    def __init__(self, z_dim=64, in_channels=3, struct_dim=128, act="relu"):
        super(Q_CelebA, self).__init__()
        self.act, self.act_inp = act_select(act)

        def conv_ln_act(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, 2, 2),
                nn.InstanceNorm2d(out_channels, affine=True),
                self.act)
        self.ls = nn.Sequential(
            nn.Conv2d(in_channels, struct_dim, 5, 2, 2),
            self.act_inp,
            conv_ln_act(struct_dim, struct_dim * 2),
            conv_ln_act(struct_dim * 2, struct_dim * 4),
            conv_ln_act(struct_dim * 4, struct_dim * 8))
        self.fc1 = nn.Linear(4 * 4 * 8 * struct_dim, 2 * z_dim)
        self.fc2 = nn.Linear(2 * z_dim, z_dim)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.act_inp(y)
        y = self.fc2(y)
        return y

# Encoder with rank restriction
class Qrank_CelebA(nn.Module):
    def __init__(self, z_dim=64, rank_min=16, in_channels=3, struct_dim=128, act="relu"):
        super(Qrank_CelebA, self).__init__()
        self.rank_offset = rank_min - 1
        self.max_id = z_dim - self.rank_offset
        self.act, self.act_inp = act_select(act)

        def conv_ln_act(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, 2, 2),
                nn.InstanceNorm2d(out_channels, affine=True),
                self.act)
        self.ls = nn.Sequential(
            nn.Conv2d(in_channels, struct_dim, 5, 2, 2),
            self.act_inp,
            conv_ln_act(struct_dim, struct_dim * 2),
            conv_ln_act(struct_dim * 2, struct_dim * 4),
            conv_ln_act(struct_dim * 4, struct_dim * 8))
        self.fc1 = nn.Linear(4 * 4 * 8 * struct_dim + self.max_id, 2 * z_dim)
        self.fc2 = nn.Linear(2 * z_dim + self.max_id, z_dim)

    def forward(self, x, rank: int):
        # id: 1, 2, ..., max_id
        id = rank - self.rank_offset
        cond = torch.zeros(x.shape[0], self.max_id, device=x.device)
        cond[:, id - 1] = 1.0
        y = self.ls(x)
        y = y.view(y.size(0), -1)
        y = torch.cat((y, cond), dim=-1)
        y = self.fc1(y)
        y = self.act_inp(y)
        y = torch.cat((y, cond), dim=-1)
        y = self.fc2(y)
        y[:, rank:] = 0.0
        return y

# Generator
class G_CelebA(nn.Module):
    def __init__(self, z_dim=64, struct_dim=128, act="relu"):
        super(G_CelebA, self).__init__()
        _, act = act_select(act)

        def dconv_act(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 5, 2, padding=2, output_padding=1, bias=False),
                # nn.BatchNorm2d(out_channels),
                act)
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, 4 * z_dim),
            act,
            nn.Linear(4 * z_dim, struct_dim * 8 * 4 * 4, bias=False),
            # nn.BatchNorm1d(struct_dim * 8 * 4 * 4),
            act)
        self.l2_5 = nn.Sequential(
            dconv_act(struct_dim * 8, struct_dim * 4),
            dconv_act(struct_dim * 4, struct_dim * 2),
            dconv_act(struct_dim * 2, struct_dim),
            nn.ConvTranspose2d(struct_dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

# Discriminator
class D_CelebA(nn.Module):
    def __init__(self, z_dim=64, in_channels=3, struct_dim=64, act="relu"):
        super(D_CelebA, self).__init__()
        act, _ = act_select(act)

        def conv_ln_act(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, 2, 2),
                nn.InstanceNorm2d(out_channels, affine=True),
                act)
        self.ls = nn.Sequential(
            nn.Conv2d(in_channels, struct_dim, 5, 2, 2),
            act,
            # nn.Dropout2d(p=0.1),
            conv_ln_act(struct_dim, struct_dim * 2),
            # nn.Dropout2d(p=0.1),
            conv_ln_act(struct_dim * 2, struct_dim * 4),
            # nn.Dropout2d(p=0.1),
            conv_ln_act(struct_dim * 4, struct_dim * 8),
            # nn.Dropout2d(p=0.1),
            nn.Conv2d(struct_dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        return y.view(-1)

# Discriminator with rank information
class Drank_CelebA(nn.Module):
    def __init__(self, z_dim=64, rank_min=16, in_channels=3, struct_dim=64, act="relu"):
        super(Drank_CelebA, self).__init__()
        self.z_dim = z_dim
        self.rank_offset = rank_min - 1
        self.max_id = z_dim - self.rank_offset
        act, self.act_inp = act_select(act)

        def conv_ln_act(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, 2, 2),
                nn.InstanceNorm2d(out_channels, affine=True),
                act)
        self.ls = nn.Sequential(
            nn.Conv2d(in_channels, struct_dim, 5, 2, 2),
            act,
            # nn.Dropout2d(p=0.1),
            conv_ln_act(struct_dim, struct_dim * 2),
            # nn.Dropout2d(p=0.1),
            conv_ln_act(struct_dim * 2, struct_dim * 4),
            # nn.Dropout2d(p=0.1),
            conv_ln_act(struct_dim * 4, struct_dim * 8),
            # nn.Dropout2d(p=0.1),
            nn.Conv2d(struct_dim * 8, 2 * z_dim, 4))
        self.fc1 = nn.Linear(2 * z_dim + self.max_id, z_dim)
        self.fc2 = nn.Linear(z_dim + self.max_id, 1)

    def forward(self, x, rank: int):
        # id: 1, 2, ..., max_id
        id = rank - self.rank_offset
        cond = torch.zeros(x.shape[0], self.max_id, device=x.device)
        cond[:, :(id - 1)] = 1.0
        y = self.ls(x)
        y = y.view(-1, 2 * self.z_dim)
        y = torch.cat((y, cond), dim=-1)
        y = self.fc1(y)
        y = self.act_inp(y)
        y = torch.cat((y, cond), dim=-1)
        y = self.fc2(y)
        return y.view(-1)

# Discriminator for Z variable
class Dz_CelebA(nn.Module):
    def __init__(self, z_dim=64, act="relu"):
        super(Dz_CelebA, self).__init__()
        _, act = act_select(act)

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 2 * z_dim),
            act,
            nn.Linear(2 * z_dim, 4 * z_dim),
            act,
            nn.Linear(4 * z_dim, 2 * z_dim),
            act,
            nn.Linear(2 * z_dim, z_dim),
            act,
            nn.Linear(z_dim, 1))

    def forward(self, x):
        out = self.fc(x).view(-1)
        return out
