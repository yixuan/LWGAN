import os
import numpy as np
import json
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
import torchvision
from torchvision.utils import make_grid
import matplotlib.pylab as plt

from utils.tools import *
from models.blocks import gen_noise_with_rank
from models.mnist import Qrank_MNIST, G_MNIST

# The nrow parameter in make_grid() is a bit misleading
# I like to use nrow to represent "number of rows"
# In make_grid() it means number of images in each row
def tensor2img(x, nrow=6, ncol=6, byrow=True):
    nimg = nrow * ncol
    xsamp = x[:nimg].view(-1, 784)
    if not byrow:
        xsamp = xsamp.view(ncol, nrow, 784).permute(1, 0, 2)
    img = make_grid(xsamp.reshape(-1, 1, 28, 28), nrow=ncol)[0, :, :]
    img = img.detach().cpu().numpy()
    return img

# Load datasets -- one digit
set_random_seed(123)
n = 6000
bsize = 256
z_dim = 16
dev = torch.device("cpu")

# Main function to generate plots
def digit_plots(digit):
    # Load data
    train_loader, test_loader = mnist_loader(
        batch_size=bsize, shuffle_train=True, shuffle_test=True,
        digit=digit, test_batch_size=bsize, device=dev)
    data = iter(train_loader)
    x = next(data)

    # True data
    # torchvision.utils.save_image(x[:36], f"./plots/mnist_digit{digit}.pdf", nrow=6)
    fig = plt.figure(figsize=(5, 6))
    img = tensor2img(x, nrow=6, ncol=6)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("True Sample: X", fontsize=20, y=1.02)
    # https://stackoverflow.com/a/4066599
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95)
    # plt.show()
    # Figure 3 in the main article
    plt.savefig(f"./plots/mnist_digit{digit}.pdf", bbox_inches=None)
    plt.close()

    # Paths to saved models
    FN = "Z_1_16_SD_64_EP_50k_BS_256_LR_1_WD_0_SC_1_IT_1_5_LG_5-0_LM_1-0_LR_0-002"
    DIR = os.path.join("./outputs/", f"MNIST_digit_{digit}", FN)
    print(DIR)

    # Load models
    with open(DIR + "/rank.json", mode="r") as f:
        rank_info = json.load(f)
    lambda_rank = rank_info["lam_est"]
    netQ = Qrank_MNIST(z_dim=z_dim, rank_min=1, struct_dim=64)
    netQ.load_state_dict(torch.load(f"{DIR}/netQ.pt", weights_only=True))
    netQ.eval()
    netG = G_MNIST(z_dim=z_dim, struct_dim=64)
    netG.load_state_dict(torch.load(f"{DIR}/netG.pt", weights_only=True))
    netG.eval()

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
    plt.xticks(summ["rank"])
    plt.xlabel(r"$s$: Rank of $A$", fontsize=20)
    plt.title(r"Rank score $\hat{\varrho}(s)$", fontsize=20, y=1.01)
    # plt.show()
    # Figure 3 in the main article
    plt.savefig(f"./plots/mnist_digit{digit}_rank.pdf", bbox_inches="tight")
    plt.close()

    # Generation
    with torch.no_grad():
        noise = gen_noise_with_rank(36, z_dim, rank, dev)
        fd = netG(noise)
    # torchvision.utils.save_image(fd, "plots/mnist_digit1_gen.pdf", nrow=6)
    fig = plt.figure(figsize=(5, 6))
    img = tensor2img(fd, nrow=6, ncol=6)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Generated sample: G(Z)", fontsize=20, y=1.02)
    # https://stackoverflow.com/a/4066599
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95)
    # plt.show()
    # Figure 3 in the main article
    plt.savefig(f"./plots/mnist_digit{digit}_gen.pdf", bbox_inches=None)
    plt.close()

    # Reconstruction
    with torch.no_grad():
        z_hat = netQ(x, rank)
        recon = netG(z_hat)
    # torchvision.utils.save_image(recon[:36], f"./plots/mnist_digit{digit}_recon.pdf", nrow=6)
    fig = plt.figure(figsize=(5, 6))
    img = tensor2img(recon, nrow=6, ncol=6)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Reconstructed sample: G(Q(X))", fontsize=20, y=1.02)
    # https://stackoverflow.com/a/4066599
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95)
    # plt.show()
    # Figure 3 in the main article
    plt.savefig(f"./plots/mnist_digit{digit}_recon.pdf", bbox_inches=None)
    plt.close()

digit_plots(digit=1)
digit_plots(digit=2)



# Load datasets -- all digits
set_random_seed(123)
n = 60000
bsize = 256
z_dim = 20
dev = torch.device("cpu")

train_loader, test_loader = mnist_loader(
    batch_size=bsize, shuffle_train=True, shuffle_test=True,
    digit=None, test_batch_size=bsize, device=dev)
data = iter(train_loader)
x = next(data)

# True data
fig = plt.figure(figsize=(5, 6))
sub = fig.add_subplot(111)
img = tensor2img(x, nrow=8, ncol=8)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("True Sample: X", fontsize=20, y=1.02)
# https://stackoverflow.com/a/4066599
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95)
# plt.show()
# Figure 4 in the main article
plt.savefig(f"plots/mnist.pdf", bbox_inches=None)
plt.close()

# Paths to saved models
FN = "Z_1_20_SD_64_EP_50k_BS_256_LR_1_WD_0_SC_1_IT_3_10_LG_5-0_LM_1-0_LR_0-0005"
DIR = os.path.join("./outputs/", "MNIST", FN)
print(DIR)

# Load models
with open(DIR + "/rank.json", mode="r") as f:
    rank_info = json.load(f)
lambda_rank = rank_info["lam_est"]
netQ = Qrank_MNIST(z_dim=z_dim, rank_min=1, struct_dim=64)
netQ.load_state_dict(torch.load(f"{DIR}/netQ.pt", weights_only=True))
netQ.eval()
netG = G_MNIST(z_dim=z_dim, struct_dim=64)
netG.load_state_dict(torch.load(f"{DIR}/netG.pt", weights_only=True))
netG.eval()

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
plt.xticks(summ["rank"])
plt.xlabel(r"$s$: Rank of $A$", fontsize=20)
plt.title(r"Rank score $\hat{\varrho}(s)$", fontsize=20, y=1.01)
# plt.show()
# Figure 4 in the main article
plt.savefig(f"./plots/mnist_rank.pdf", bbox_inches="tight")
plt.close()

# Generation
# rank = 1
with torch.no_grad():
    noise = gen_noise_with_rank(64, z_dim, rank, dev)
    fd = netG(noise)
fig = plt.figure(figsize=(5, 6))
img = tensor2img(fd, nrow=8, ncol=8)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Generated sample: G(Z)", fontsize=20, y=1.02)
# https://stackoverflow.com/a/4066599
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95)
# plt.show()
# Figure 4 in the main article
plt.savefig(f"./plots/mnist_gen.pdf", bbox_inches=None)
plt.close()

# Reconstruction
with torch.no_grad():
    z_hat = netQ(x, rank)
    recon = netG(z_hat)
fig = plt.figure(figsize=(5, 6))
img = tensor2img(recon, nrow=8, ncol=8)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Reconstructed sample: G(Q(X))", fontsize=20, y=1.02)
# https://stackoverflow.com/a/4066599
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95)
# plt.show()
# Figure 4 in the main article
plt.savefig(f"./plots/mnist_recon.pdf", bbox_inches=None)
plt.close()

# Interpolation
npair = 18
ninterp = 8
start = z_hat[:npair]
end = z_hat[npair:(2 * npair)]
diff = end - start
res = []
for i in range(ninterp):
    inter = start + i / float(ninterp - 1) * diff
    res.append(netG(inter))
res = torch.concat(res, dim=0)

fig = plt.figure(figsize=(11, 6))
img = tensor2img(res, nrow=ninterp, ncol=npair)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Interpolation: Top to Bottom", fontsize=20, y=1.02)
# https://stackoverflow.com/a/4066599
plt.subplots_adjust(left=0.04, bottom=0.06, right=0.96)
# plt.show()
# Figure 4 in the main article
plt.savefig(f"./plots/mnist_interp.pdf", bbox_inches=None)
plt.close()
