import torch
import torch.nn as nn

# Encoder with rank restriction
class Qrank_MNIST(nn.Module):
    def __init__(self, z_dim, rank_min, struct_dim=128):
        super(Qrank_MNIST, self).__init__()
        self.rank_offset = rank_min - 1
        self.max_id = z_dim - self.rank_offset
        self.features = 4 * 4 * 4 * struct_dim
        self.act = nn.LeakyReLU(0.1, inplace=True)
        # self.act = nn.ReLU(inplace=True)
        # self.act = nn.SiLU(inplace=True)
        self.main = nn.Sequential(
            nn.Conv2d(1, struct_dim, 5, stride=2, padding=2),
            self.act,
            nn.Conv2d(struct_dim, 2 * struct_dim, 5, stride=2, padding=2),
            self.act,
            nn.Conv2d(2 * struct_dim, 4 * struct_dim, 5, stride=2, padding=2),
            self.act)
        self.fc1 = nn.Linear(self.features + self.max_id, 2 * z_dim)
        self.fc2 = nn.Linear(2 * z_dim + self.max_id, z_dim)

    def forward(self, input, rank: int):
        # id: 1, 2, ..., max_id
        id = rank - self.rank_offset
        cond = torch.zeros(input.shape[0], self.max_id, device=input.device)
        cond[:, id - 1] = 1.0
        out = self.main(input)
        out = out.view(-1, self.features)
        out = torch.cat((out, cond), dim=-1)
        out = self.fc1(out)
        out = self.act(out)
        out = torch.cat((out, cond), dim=-1)
        out = self.fc2(out)
        out[:, rank:] = 0.0
        return out

# Generator
class G_MNIST(nn.Module):
    def __init__(self, z_dim, struct_dim=128):
        super(G_MNIST, self).__init__()
        self.struct_dim = struct_dim
        self.act = nn.LeakyReLU(0.1, inplace=True)
        # self.act = nn.ReLU(inplace=True)
        # self.act = nn.SiLU(inplace=True)
        preprocess = nn.Sequential(
            nn.Linear(z_dim, 4 * z_dim),
            self.act,
            nn.Linear(4 * z_dim, 4 * 4 * 4 * struct_dim),
            self.act
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * struct_dim, 2 * struct_dim, 5),
            self.act
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * struct_dim, struct_dim, 5),
            self.act
        )
        deconv_out = nn.ConvTranspose2d(struct_dim, 1, 8, stride=2)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.struct_dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, 1, 28, 28)

# Discriminator
class Drank_MNIST(nn.Module):
    def __init__(self, z_dim, rank_min, struct_dim=128):
        super(Drank_MNIST, self).__init__()
        self.rank_offset = rank_min - 1
        self.max_id = z_dim - self.rank_offset
        self.features = 4 * 4 * 4 * struct_dim
        self.act = nn.LeakyReLU(0.1, inplace=True)
        # self.act = nn.ReLU(inplace=True)
        # self.act = nn.SiLU(inplace=True)
        main = nn.Sequential(
            nn.Conv2d(1, struct_dim, 5, stride=2, padding=2),
            self.act,
            nn.Conv2d(struct_dim, 2 * struct_dim, 5, stride=2, padding=2),
            self.act,
            nn.Conv2d(2 * struct_dim, 4 * struct_dim, 5, stride=2, padding=2),
            self.act
        )
        self.main = main
        # self.output = nn.Linear(self.features, 1)
        self.fc1 = nn.Linear(self.features + self.max_id, 2 * z_dim)
        self.fc2 = nn.Linear(2 * z_dim + self.max_id, 1)

    def forward(self, input, rank: int):
        # id: 1, 2, ..., max_id
        id = rank - self.rank_offset
        cond = torch.zeros(input.shape[0], self.max_id, device=input.device)
        cond[:, :(id - 1)] = 1.0
        out = self.main(input)
        out = out.view(-1, self.features)
        out = torch.cat((out, cond), dim=-1)
        out = self.fc1(out)
        out = self.act(out)
        out = torch.cat((out, cond), dim=-1)
        out = self.fc2(out)
        return out.view(-1)
        # out = self.main(input)
        # out = out.view(-1, self.features)
        # out = self.output(out)
        # return out.view(-1)
