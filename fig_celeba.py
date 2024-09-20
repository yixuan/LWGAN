import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import numpy as np
import json
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
import seaborn as sns

from utils.tools import *
from models.blocks import mask_noise
from models.celeba import Qrank_CelebA, Q_CelebA, G_CelebA

# The directory containing face64.pt
data_dir = "./data/celeba/"

# Load datasets
set_random_seed(123)
bsize = 256
z_dim = 128
x_dim = 64 * 64 * 3
npair = 6
ninterp = 10
dev = torch.device("cpu")
face64 = torch.load(
    os.path.join(data_dir, "face64.pt"),
    map_location=dev, weights_only=True)
train_loader = DataLoader(face64, batch_size=bsize, shuffle=False, drop_last=True)
data = iter(train_loader)
n = face64.shape[0]  # 16055
x = next(data)
x1 = x[:npair]
x2 = x[npair:(2 * npair)]

# The nrow parameter in make_grid() is a bit misleading
# I like to use nrow to represent "number of rows"
# In make_grid() it means number of images in each row
def tensor2img(x, nrow=6, ncol=6, byrow=True):
    nimg = nrow * ncol
    xsamp = 0.5 * (1.0 + x[:nimg].view(-1, x_dim))
    if not byrow:
        xsamp = xsamp.view(ncol, nrow, x_dim).permute(1, 0, 2)
    img = make_grid(xsamp.reshape(-1, 3, 64, 64), nrow=ncol)
    img = img.detach().cpu().numpy().transpose(1, 2, 0)
    return img

# True data
fig = plt.figure(figsize=(8, 6))
img = tensor2img(x, nrow=6, ncol=10)
plt.imshow(img)
plt.axis("off")
plt.title("True Sample", fontsize=20, y=1.02)
# https://stackoverflow.com/a/4066599
plt.subplots_adjust(left=0.03, bottom=0.05, right=0.97)
# plt.show()
# Figure 5(a) in the main article
plt.savefig(f"./plots/celeba_wide.jpg", bbox_inches=None)
plt.close()

fig = plt.figure(figsize=(6, 6))
img = tensor2img(x, nrow=6, ncol=6)
plt.imshow(img)
plt.axis("off")
plt.title("True Sample", fontsize=20, y=1.02)
# plt.show()
# Figure 7 in the main article
plt.savefig(f"./plots/celeba.jpg", bbox_inches="tight")
plt.close()

def gen_img(G, z_dim, bsize, rank=None):
    G.eval()
    with torch.no_grad():
        noise = torch.randn(bsize, z_dim, device=dev)
        if rank is not None:
            noise = mask_noise(noise, rank)
    img = tensor2img(G(noise), nrow=6, ncol=6)
    return img

def recon_img(G, Q, x, rank=None):
    G.eval()
    Q.eval()
    with torch.no_grad():
        if rank is not None:
            recon = G(Q(x, rank))
        else:
            recon = G(Q(x))
    img = tensor2img(recon, nrow=6, ncol=6)
    return img

def interp_img(G, Q, x1, x2, npair=6, ninterp=10, rank=None):
    G.eval()
    Q.eval()
    with torch.no_grad():
        if rank is not None:
            z1, z2 = Q(x1[:npair], rank), Q(x2[:npair], rank)
        else:
            z1, z2 = Q(x1[:npair]), Q(x2[:npair])
    diff = z2 - z1
    res = []
    for i in range(ninterp):
        inter = z1 + i / float(ninterp - 1) * diff
        res.append(netG(inter))
    res = torch.concat(res, dim=0)
    img = tensor2img(res, nrow=npair, ncol=ninterp, byrow=False)
    return img

# WGAN
torch.manual_seed(123)
config = "Z_1_128_SD_64_EP_100k_BS_128_LR_2_WD_0_SC_1_IT_1_10_LG_20-0_LM_0-0_LR_0-01"
path = f"./outputs/WGAN/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, weights_only=True))
netG.eval()
# Generation
img = gen_img(netG, z_dim, bsize)
fig = plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Generated sample: WGAN", fontsize=20, y=1.02)
# plt.show()
# Figure 6 in the main article
plt.savefig(f"./plots/celeba_gen_wgan.jpg", bbox_inches="tight")
plt.close()

# WAE
torch.manual_seed(123)
config = "Z_1_128_SD_64_EP_100k_BS_128_LR_5_WD_0_SC_1_IT_1_10_LG_5-0_LM_100-0_LR_0-01"
path = f"./outputs/WAE/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, weights_only=True))
netG.eval()
path = f"./outputs/WAE/CelebA/{config}/netQ.pt"
netQ = Q_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netQ.load_state_dict(torch.load(path, weights_only=True))
netQ.eval()
# Generation
img = gen_img(netG, z_dim, bsize)
fig = plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Generated sample: WAE", fontsize=20, y=1.02)
# plt.show()
# Figure 6 in the main article
plt.savefig(f"./plots/celeba_gen_wae.jpg", bbox_inches="tight")
plt.close()
# Reconstruction
img = recon_img(netG, netQ, x)
fig = plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Reconstruction: WAE", fontsize=20, y=1.02)
# plt.show()
# Figure 7 in the main article
plt.savefig(f"./plots/celeba_recon_wae.jpg", bbox_inches="tight")
plt.close()
# Interpolation
img = interp_img(netG, netQ, x1, x2, npair, ninterp)
fig = plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Interpolation: WAE", fontsize=20, y=1.02)
# plt.show()
# Figure 8 in the main article
plt.savefig(f"./plots/celeba_interp_wae.jpg", bbox_inches="tight")
plt.close()

# CycleGAN
torch.manual_seed(123)
config = "Z_1_128_SD_64_EP_10k_BS_128_LR_1_WD_0_SC_1_IT_5_15_LG_5-0_LM_0-0_LR_0-01"
path = f"./outputs/CycleGAN/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, weights_only=True))
netG.eval()
path = f"./outputs/CycleGAN/CelebA/{config}/netQ.pt"
netQ = Q_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netQ.load_state_dict(torch.load(path, weights_only=True))
netQ.eval()
# Generation
img = gen_img(netG, z_dim, bsize)
fig = plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Generated sample: CycleGAN", fontsize=20, y=1.02)
# plt.show()
# Figure 6 in the main article
plt.savefig(f"./plots/celeba_gen_cyclegan.jpg", bbox_inches="tight")
plt.close()
# Reconstruction
img = recon_img(netG, netQ, x)
fig = plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Reconstruction: CycleGAN", fontsize=20, y=1.02)
# plt.show()
# Figure 7 in the main article
plt.savefig(f"./plots/celeba_recon_cyclegan.jpg", bbox_inches="tight")
plt.close()
# Interpolation
img = interp_img(netG, netQ, x1, x2, npair, ninterp)
fig = plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis("off")
plt.title("Interpolation: CycleGAN", fontsize=20, y=1.02)
# plt.show()
# Figure 8 in the main article
plt.savefig(f"./plots/celeba_interp_cyclegan.jpg", bbox_inches="tight")
plt.close()

# LWGAN
torch.manual_seed(123)
config = "Z_1_128_SD_64_EP_100k_BS_128_LR_1_WD_0_SC_1_IT_5_10_LG_5-0_LM_1-0_LR_0-002"
DIR = os.path.join("./outputs/", "CelebA", config)
print(DIR)
path = f"./outputs/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
netG.eval()
path = f"./outputs/CelebA/{config}/netQ.pt"
netQ = Qrank_CelebA(z_dim=z_dim, rank_min=1, struct_dim=64, act="relu").to(device=dev)
netQ.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
netQ.eval()
with open(DIR + "/rank.json", mode="r") as f:
    rank_info = json.load(f)
lambda_rank = rank_info["lam_est"]

# Estimated rank
summ = pd.read_csv(f"{DIR}/summ.csv")
rho = summ["critic_loss"] + summ["recon_error"] + \
      summ["grad_penalty"] + summ["mmd"] + \
      summ["rank"] * lambda_rank
rho = rho.values
ind = np.argmin(rho)
rank = int(summ["rank"].iloc[ind])
score = rho[ind]
fig = plt.figure(figsize=(6, 6))
plt.plot(summ["rank"], rho)
# https://stackoverflow.com/a/17432641
plt.scatter(rank, score, marker="*", c="red", s=200, zorder=2)
# plt.xticks(summ["rank"])
plt.xlabel(r"$s$: Rank of $A$", fontsize=20)
plt.title(r"Rank score $\hat{\varrho}(s)$", fontsize=20, y=1.01)
# plt.show()
# Figure 5(b) in the main article
plt.savefig(f"./plots/celeba_rank.pdf", bbox_inches="tight")
plt.close()

# Generation
for r in [rank, 16, 32, 64, 128]:
    torch.manual_seed(123)
    img = gen_img(netG, z_dim, bsize, rank=r)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Generated sample: LWGAN ({r})", fontsize=20, y=1.02)
    # plt.show()
    # Figure 6 in the main article
    plt.savefig(f"./plots/celeba_gen_lwgan{r}.jpg", bbox_inches="tight")
    plt.close()

# Reconstruction
for r in [rank, 16, 32, 64, 128]:
    torch.manual_seed(123)
    img = recon_img(netG, netQ, x, rank=r)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Reconstruction: LWGAN ({r})", fontsize=20, y=1.02)
    # plt.show()
    # Figure 7 in the main article
    plt.savefig(f"./plots/celeba_recon_lwgan{r}.jpg", bbox_inches="tight")
    plt.close()

# Interpolation
img = interp_img(netG, netQ, x1, x2, npair, ninterp, rank=rank)
fig = plt.figure(figsize=(10, 6))
sub = fig.add_subplot(111)
plt.imshow(img)
plt.axis("off")
plt.title(f"Interpolation: LWGAN ({rank})", fontsize=20, y=1.02)
# plt.show()
# Figure 8 in the main article
plt.savefig(f"./plots/celeba_interp_lwgan{rank}.jpg", bbox_inches="tight")
plt.close()

# Show the training losses
with open(DIR + "/losses.json", mode="r") as f:
    losses = json.load(f)
losses = pd.DataFrame.from_dict(losses)
losses["epoch"] = losses["epoch"] / 1000
fig = plt.figure(figsize=(18, 4))
sub = fig.add_subplot(131)
sns.lineplot(data=losses, x="epoch", y="pre_critic_loss")
plt.xlabel(r"Epoch ($\times 1000$)", fontsize=16)
plt.ylabel("")
plt.title("Pre-GQ Critic Loss", fontsize=16)
sub = fig.add_subplot(132)
sns.lineplot(data=losses, x="epoch", y="post_critic_loss")
plt.xlabel(r"Epoch ($\times 1000$)", fontsize=16)
plt.ylabel("")
plt.title("Post-GQ Critic Loss", fontsize=16)
sub = fig.add_subplot(133)
sns.lineplot(data=losses, x="epoch", y="post_recon_error")
plt.xlabel(r"Epoch ($\times 1000$)", fontsize=16)
plt.ylabel("")
plt.title("Reconstruction Error", fontsize=16)
# plt.show()
# Figure 8 in the main article
plt.savefig("./plots/celeba_losses.pdf", bbox_inches="tight")
plt.close()
