import torch
import torch.nn as nn

class CycleGAN(nn.Module):
    def __init__(self, z_dim, netQ, netG, netD, netDz, device=torch.device("cpu")):
        super(CycleGAN, self).__init__()
        self.z_dim = z_dim
        self.netQ = netQ
        self.netG = netG
        self.netD = netD
        self.netDz = netDz
        self.device = device

    # Loss function for discriminators (D and Dz)
    @torch.jit.export
    def D_loss(self, x, z):
        # Adversarial loss on X
        Gz = self.netG(z)
        Dx = self.netD(x).sigmoid_()
        DGz = self.netD(Gz).sigmoid_()
        loss_D = (Dx - 1.0).square().mean() + DGz.square().mean()
        # Adversarial loss on Z
        Qx = self.netQ(x)
        Dzz = self.netDz(z).sigmoid_()
        DzQx = self.netDz(Qx).sigmoid_()
        loss_Dz = (Dzz - 1.0).square().mean() + DzQx.square().mean()
        return loss_D + loss_Dz

    # Loss function for G and Q
    @torch.jit.export
    def GQ_loss(self, x, z, lambda_cycle: float):
        # GAN loss
        Gz = self.netG(z)
        DGz = self.netD(Gz).sigmoid_()
        Qx = self.netQ(x)
        DzQx = self.netDz(Qx).sigmoid_()
        loss_gan = (DGz - 1.0).square().mean() + (DzQx - 1.0).square().mean()
        # Cycle consistency loss
        x_recon = self.netG(self.netQ(x))
        z_recon = self.netQ(self.netG(z))
        loss_cyc = (x_recon - x).abs().mean() + (z_recon - z).abs().mean()
        return loss_gan + lambda_cycle * loss_cyc

    # Use GQ loss for forward pass
    @torch.jit.export
    def forward(self, x, z, lambda_cycle: float):
        return self.GQ_loss(x, z, lambda_cycle)
