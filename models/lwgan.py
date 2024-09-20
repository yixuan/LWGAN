import torch
import torch.nn as nn

from models.blocks import gen_noise_with_rank
from models.losses import _gradient_penalty, mmd_penalty

class LWGAN(nn.Module):
    def __init__(self, z_dim, netQ, netG, netD, device=torch.device("cpu")):
        super(LWGAN, self).__init__()
        self.z_dim = z_dim
        self.netQ = netQ
        self.netG = netG
        self.netD = netD
        self.device = device

    @torch.jit.export
    def D_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        post_data = self.netG(self.netQ(real_data, rank))
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = -torch.abs(diff) if abs else -diff
        return losses.mean()

    # Loss function for G and Q update
    @torch.jit.export
    def GQ_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = l2 + torch.abs(diff) if abs else l2 + diff
        return losses.mean()

    # Gradient penalty for netD
    @torch.jit.ignore
    def gradient_penalty_D(self, x, z, rank: int):
        x_hat = self.netG(z)
        x_tilde = self.netG(self.netQ(x, rank))
        return _gradient_penalty(x_hat, x_tilde, lambda x: self.netD(x, rank))

    # MMD penalty
    @torch.jit.export
    def mmd_penalty(self, real_data, rank: int, lambda_mmd: float):
        n = real_data.shape[0]
        # MMD
        mmd = torch.tensor([0.0], device=real_data.device)
        if lambda_mmd != 0.0:
            z = gen_noise_with_rank(n, self.z_dim, rank, self.device)
            z_hat = self.netQ(real_data, rank)
            mmd = lambda_mmd * mmd_penalty(z_hat, z, kernel="IMQ", sigma2_p=1.0)
        return mmd

    @torch.jit.ignore
    def dual_loss(self, x1, x2, rank: int, lambda_gp: float):
        n = x1.shape[0]
        noise = gen_noise_with_rank(n, self.z_dim, rank, self.device)
        fake_data = self.netG(noise)
        # Loss function
        cost_D = self.D_loss(x1, fake_data, rank, abs=False)
        # Gradient penalty
        z = gen_noise_with_rank(n, self.z_dim, rank, self.device)
        gp_D = self.gradient_penalty_D(x2.data, z.data, rank)
        # Dual loss
        dual_cost = cost_D + lambda_gp * gp_D
        return dual_cost

    # Reconstruction loss
    @torch.jit.export
    def recon_loss(self, real_data, rank: int):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        return l2.mean()

    # Primal loss
    @torch.jit.export
    def forward(self, x1, x2, rank: int, lambda_mmd: float, lambda_rank: float):
        n = x1.shape[0]
        noise = gen_noise_with_rank(n, self.z_dim, rank, self.device)
        fake_data = self.netG(noise)
        # Loss function
        cost_GQ = self.GQ_loss(x1, fake_data, rank, abs=False)
        # MMD
        mmd = self.mmd_penalty(x2, rank, lambda_mmd)
        # Primal loss
        primal_cost = cost_GQ + mmd + lambda_rank * rank
        return primal_cost
