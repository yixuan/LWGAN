import torch
import torch.nn as nn

from models.blocks import gen_noise
from models.losses import mmd_penalty

class WAE(nn.Module):
    def __init__(self, z_dim, netQ, netG, device=torch.device("cpu")):
        super(WAE, self).__init__()
        self.z_dim = z_dim
        self.netQ = netQ
        self.netG = netG
        self.device = device

    # Reconstruction loss
    @torch.jit.export
    def recon_loss(self, real_data):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        return l2.mean()

    # MMD penalty
    @torch.jit.export
    def mmd_penalty(self, real_data, lambda_mmd: float):
        n = real_data.shape[0]
        # MMD
        mmd = torch.tensor([0.0], device=real_data.device)
        if lambda_mmd != 0.0:
            z = gen_noise(n, self.z_dim, self.device)
            z_hat = self.netQ(real_data)
            mmd = lambda_mmd * mmd_penalty(z_hat, z, kernel="IMQ", sigma2_p=1.0)
        return mmd

    # Total loss
    @torch.jit.export
    def forward(self, x1, x2, lambda_mmd: float):
        # Reconstruction error
        recon = self.recon_loss(x1)
        # MMD penalty
        mmd = self.mmd_penalty(x2, lambda_mmd)
        # Total loss
        loss = recon + mmd
        return loss
