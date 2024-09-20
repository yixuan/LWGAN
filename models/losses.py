import torch
from torch.autograd import grad

# Gradient penalty function
def _gradient_penalty(x, y, f):
    dev = x.device
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    # Interpolation
    alpha = torch.rand(shape, device=dev)
    z = x + alpha * (y - x)
    z.requires_grad_(True)
    # Gradient penalty
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones_like(o), create_graph=True)[0].view(z.size(0), -1)
    gp = (torch.square(g.norm(p=2, dim=1) - 1.0)).mean()
    return gp

# *** MMD penalty ***
# MMD loss between z and Q(x)
@torch.jit.script
def mmd_penalty(z_hat, z, kernel: str = "RBF", sigma2_p: float = 1.0):
    n = z.shape[0]
    zdim = z.shape[1]
    half_size = int((n * n - n) / 2)

    # norms_z = z.square().sum(dim=1, keepdim=True)
    # dots_z = torch.mm(z, z.t())
    # dists_z = (norms_z + norms_z.t() - 2.0 * dots_z).abs()
    dists_z = torch.cdist(z, z).square()

    # norms_zh = z_hat.square().sum(dim=1, keepdim=True)
    # dots_zh = torch.mm(z_hat, z_hat.t())
    # dists_zh = (norms_zh + norms_zh.t() - 2.0 * dots_zh).abs()
    dists_zh = torch.cdist(z_hat, z_hat).square()

    # dots = torch.mm(z_hat, z.t())
    # dists = (norms_zh + norms_z.t() - 2.0 * dots).abs()
    dists = torch.cdist(z_hat, z).square()

    stat = torch.tensor([0.0], device=z.device)
    if kernel == "RBF":
        sigma2_k = torch.topk(dists.reshape(-1), half_size)[0][-1]
        sigma2_k = sigma2_k + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]
        res1 = torch.exp(-dists_zh / 2.0 / sigma2_k)
        res1 = res1 + torch.exp(-dists_z / 2.0 / sigma2_k)
        res1 = torch.mul(res1, 1.0 - torch.eye(n, device=z.device))
        res1 = res1.sum() / (n * n - n)
        res2 = torch.exp(-dists / 2.0 / sigma2_k)
        res2 = res2.sum() * 2.0 / (n * n)
        stat = res1 - res2
    elif kernel == "IMQ":
        Cbase = 2.0 * zdim * sigma2_p
        mask = 1.0 - torch.eye(n, device=z.device)
        for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
            C = Cbase * scale
            res1 = C / (C + dists_z) + C / (C + dists_zh)
            res1 = torch.mul(res1, mask)
            res1 = res1.sum() / (n * n - n)
            res2 = C / (C + dists)
            res2 = res2.sum() * 2.0 / (n * n)
            stat = stat + res1 - res2
    return stat
