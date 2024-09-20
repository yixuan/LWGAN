import torch
import torch.nn as nn

from models.blocks import gen_noise
from models.losses import _gradient_penalty

class WGAN(nn.Module):
    def __init__(self, z_dim, netG, netD, device=torch.device("cpu")):
        super(WGAN, self).__init__()
        self.z_dim = z_dim
        self.netG = netG
        self.netD = netD
        self.device = device

    # Gradient penalty for netD
    @torch.jit.ignore
    def gradient_penalty_D(self, x, z):
        x_hat = self.netG(z)
        return _gradient_penalty(x_hat, x, self.netD)

    # Critic difference
    @torch.jit.export
    def critic_diff(self, x):
        # Generate fake data
        n = x.shape[0]
        noise = gen_noise(n, self.z_dim, self.device)
        fake_data = self.netG(noise)
        # Critic difference
        diff = self.netD(x) - self.netD(fake_data)
        # Critic maximizes diff, so the loss takes a negative sign
        return -diff.mean()

    # Loss function for netD
    @torch.jit.ignore
    def D_loss(self, x1, x2, lambda_gp: float):
        # Critic difference
        loss_diff = self.critic_diff(x1)
        # Gradient penalty
        n = x2.shape[0]
        z = gen_noise(n, self.z_dim, self.device)
        gp = self.gradient_penalty_D(x2.data, z.data)
        # Total loss
        return loss_diff + lambda_gp * gp

    # Loss function for G update
    @torch.jit.export
    def G_loss(self, x):
        # Generate fake data
        n = x.shape[0]
        noise = gen_noise(n, self.z_dim, self.device)
        fake_data = self.netG(noise)
        # Critic difference
        # diff = self.netD(x) - self.netD(fake_data)
        # We only optimize G, so the first term is not needed
        loss = -self.netD(fake_data)
        return loss.mean()

    # Use G loss for forward pass
    @torch.jit.export
    def forward(self, x):
        return self.G_loss(x)
