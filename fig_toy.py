import os
import numpy as np
import json
from scipy import stats
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
import matplotlib.pylab as plt
import seaborn as sns

from utils.tools import *
from models.blocks import gen_noise_with_rank
from models.toy import Qrank_Toy, G_Toy

##### Swiss roll #####

# Load datasets
set_random_seed(123)
z_dim = 5
x_dim = 2
bsize = 512
data = inf_train_gen("swissroll", batch_size=bsize, std=0.0)
rd = next(data)

# True data
fig = plt.figure(figsize=(6, 6))
plt.scatter(rd[:, 0], rd[:, 1])
plt.title("True Sample: X", fontsize=20)
plt.xlabel(r"$X_1$", fontsize=20)
plt.ylabel(r"$X_2$", fontsize=20)
# plt.show()
# Figure 2(a) in the main article
plt.savefig("./plots/swissroll.pdf", bbox_inches="tight")
plt.close()

# Paths to saved models
FN = "Z_5_QGD_64_64_64_EP_5k_BS_512_LR_2_SC_1_IT_5_20_LG_5-0_LM_1-0_LR_0-01"
DIR = os.path.join("./outputs/", "swissroll", FN)
print(DIR)

# Load models
dev = torch.device("cpu")
with open(DIR + "/rank.json", mode="r") as f:
    rank_info = json.load(f)
lambda_rank = rank_info["lam_est"]
netQ = Qrank_Toy(z_dim=z_dim, rank_min=1, input_dim=x_dim, struct_dim=64)
netQ.load_state_dict(torch.load(f"{DIR}/netQ.pt", weights_only=True))
netQ.eval()
netG = G_Toy(z_dim=z_dim, struct_dim=64, out_dim=x_dim)
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
plt.title(r"Rank score $\hat{\varrho}(s)$", fontsize=20)
# plt.show()
# Figure 2(a) in the main article
plt.savefig("./plots/swissroll_rank.pdf", bbox_inches="tight")
plt.close()

# Generation
# rank = 1
x = torch.tensor(rd)
with torch.no_grad():
    noise = gen_noise_with_rank(bsize, z_dim, rank, dev)
    fd = netG(noise).detach().cpu().numpy()
# Reconstruction
with torch.no_grad():
    z_hat = netQ(x, rank)
    recon = netG(z_hat)
z_hat = pd.DataFrame(z_hat.detach().cpu().numpy(), columns=[f"Z{i}" for i in range(1, z_dim + 1)])
recon = recon.detach().cpu().numpy()

# Latent space
fig = plt.figure(figsize=(8, 8))
sns.pairplot(data=z_hat)
plt.show()

# Observation space
# fig = plt.figure(figsize=(18, 6))
# sub = fig.add_subplot(131)
# plt.scatter(rd[:, 0], rd[:, 1])
# plt.title("True Sample: X")
# sub = fig.add_subplot(132)
# plt.scatter(fd[:, 0], fd[:, 1])
# plt.title("Generated sample: G(Z)")
# sub = fig.add_subplot(133)
# plt.scatter(recon[:, 0], recon[:, 1])
# plt.title("Reconstructed sample: G(Q(X))")
# plt.show()
fig = plt.figure(figsize=(6, 6))
plt.scatter(fd[:, 0], fd[:, 1])
plt.title("Generated sample: G(Z)", fontsize=20)
plt.xlabel(r"$X_1$", fontsize=20)
plt.ylabel(r"$X_2$", fontsize=20)
# plt.show()
# Figure 2(a) in the main article
plt.savefig("./plots/swissroll_gen.pdf", bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(6, 6))
plt.scatter(recon[:, 0], recon[:, 1])
plt.title("Reconstructed sample: G(Q(X))", fontsize=20)
plt.xlabel(r"$X_1$", fontsize=20)
plt.ylabel(r"$X_2$", fontsize=20)
# plt.show()
# Figure 2(a) in the main article
plt.savefig("./plots/swissroll_recon.pdf", bbox_inches="tight")
plt.close()



##### S-curve #####

# Load datasets
set_random_seed(123)
z_dim = 5
x_dim = 3
bsize = 512
data = inf_train_gen("scurve", batch_size=bsize, std=0.0)
rd = next(data)

# True data
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection="3d")
ax.scatter(rd[:, 0], rd[:, 1], rd[:, 2], c=rd[:, 2], cmap=plt.cm.Spectral)
ax.view_init(15, -110)
ax.set_title("True Sample: X", fontsize=20, y=0.95)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.1, 2.1)
ax.set_zlim(-2.1, 2.1)
# plt.show()
# Figure 2(b) in the main article
plt.savefig("./plots/scurve.pdf", bbox_inches="tight")
plt.close()

# Paths to saved models
FN = "Z_5_QGD_64_64_64_EP_10k_BS_512_LR_2_SC_1_IT_1_20_LG_5-0_LM_1-0_LR_0-01"
DIR = os.path.join("./outputs/", "scurve", FN)
print(DIR)

# Load models
dev = torch.device("cpu")
with open(DIR + "/rank.json", mode="r") as f:
    rank_info = json.load(f)
lambda_rank = rank_info["lam_est"]
netQ = Qrank_Toy(z_dim=z_dim, rank_min=1, input_dim=x_dim, struct_dim=64)
netQ.load_state_dict(torch.load(f"{DIR}/netQ.pt", weights_only=True))
netQ.eval()
netG = G_Toy(z_dim=z_dim, struct_dim=64, out_dim=x_dim)
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
plt.title(r"Rank score $\hat{\varrho}(s)$", fontsize=20)
# plt.show()
# Figure 2(b) in the main article
plt.savefig("./plots/scurve_rank.pdf", bbox_inches="tight")
plt.close()

# Generation
# rank = 2
x = torch.tensor(rd)
with torch.no_grad():
    noise = gen_noise_with_rank(bsize, z_dim, rank, dev)
    fd = netG(noise).detach().cpu().numpy()
# Reconstruction
with torch.no_grad():
    z_hat = netQ(x, rank)
    recon = netG(z_hat)
z_hat = pd.DataFrame(z_hat.detach().cpu().numpy(), columns=[f"Z{i}" for i in range(1, z_dim + 1)])
recon = recon.detach().cpu().numpy()

# Latent space
fig = plt.figure(figsize=(8, 8))
sns.pairplot(data=z_hat)
plt.show()

# Observation space
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection="3d")
ax.scatter(fd[:, 0], fd[:, 1], fd[:, 2], c=fd[:, 2], cmap=plt.cm.Spectral)
ax.view_init(15, -110)
ax.set_title("Generated sample: G(Z)", fontsize=20, y=0.95)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.1, 2.1)
ax.set_zlim(-2.1, 2.1)
# plt.show()
# Figure 2(b) in the main article
plt.savefig("./plots/scurve_gen.pdf", bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection="3d")
ax.scatter(recon[:, 0], recon[:, 1], recon[:, 2], c=recon[:, 2], cmap=plt.cm.Spectral)
ax.view_init(15, -110)
ax.set_title("Reconstructed sample: G(Q(X))", fontsize=20, y=0.95)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.1, 2.1)
ax.set_zlim(-2.1, 2.1)
# plt.show()
# Figure 2(b) in the main article
plt.savefig("./plots/scurve_recon.pdf", bbox_inches="tight")
plt.close()



##### Hyperplane #####

# Load datasets
set_random_seed(123)
z_dim = 7
x_dim = 5
bsize = 512
data = inf_train_gen("hyperplane", batch_size=bsize, std=0.0)
rd = next(data)

# Paths to saved models
FN = "Z_7_QGD_64_64_64_EP_10k_BS_512_LR_2_SC_1_IT_1_20_LG_5-0_LM_1-0_LR_0-01"
DIR = os.path.join("./outputs/", "hyperplane", FN)
print(DIR)

# Load models
dev = torch.device("cpu")
with open(DIR + "/rank.json", mode="r") as f:
    rank_info = json.load(f)
lambda_rank = rank_info["lam_est"]
netQ = Qrank_Toy(z_dim=z_dim, rank_min=1, input_dim=x_dim, struct_dim=64)
netQ.load_state_dict(torch.load(f"{DIR}/netQ.pt", weights_only=True))
netQ.eval()
netG = G_Toy(z_dim=z_dim, struct_dim=64, out_dim=x_dim)
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
plt.title(r"Rank score $\hat{\varrho}(s)$", fontsize=20)
# plt.show()
# Figure 2(c) in the main article
plt.savefig("./plots/hyperplane_rank.pdf", bbox_inches="tight")
plt.close()

# True data
gd = pd.DataFrame(rd, columns=[f"X{i}" for i in range(1, x_dim + 1)])
fig = plt.figure(figsize=(6, 6))
# https://stackoverflow.com/a/31214149
sns.set(style="white", font_scale=1.5, palette="tab10")
# https://stackoverflow.com/a/47200170
g = sns.pairplot(data=gd, plot_kws=dict(alpha=0.3))
# Add title
# https://stackoverflow.com/q/29813694
g.fig.subplots_adjust(top=0.93)
g.fig.suptitle("True Sample: X", fontsize=40)
# Adjust limits for each subplot
# https://stackoverflow.com/a/51292299
for i in range(5):
    for j in range(4):
        g.axes[i, j].set_xlim(-3.5, 3.5)
        g.axes[j, i].set_ylim(-3.5, 3.5)
    # The last row
    g.axes[4, i].set_ylim(-6, 10)
    # The last column
    g.axes[i, 4].set_xlim(-6, 10)
# plt.show()
# Figure 2(c) in the main article
plt.savefig("./plots/hyperplane.png", bbox_inches="tight")
plt.close()

# Generation
# rank = 4
x = torch.tensor(rd)
with torch.no_grad():
    noise = gen_noise_with_rank(bsize, z_dim, rank, dev)
    fd = netG(noise)
# Reconstruction
with torch.no_grad():
    z_hat = netQ(x, rank)
    recon = netG(z_hat)
z_hat = pd.DataFrame(z_hat.detach().cpu().numpy(), columns=[f"Z{i}" for i in range(1, z_dim + 1)])
recon = pd.DataFrame(recon.detach().cpu().numpy(), columns=[f"X{i}" for i in range(1, x_dim + 1)])
fd = pd.DataFrame(fd.detach().cpu().numpy(), columns=[f"X{i}" for i in range(1, x_dim + 1)])

# Latent space
fig = plt.figure(figsize=(8, 8))
sns.pairplot(data=z_hat)
plt.show()

# Observation space
# fig = plt.figure(figsize=(6, 6))
# sns.pairplot(data=gd)
# # plt.title("True Sample: X")
# plt.show()

fig = plt.figure(figsize=(6, 6))
sns.set(style="white", font_scale=1.5, palette="tab10")
g = sns.pairplot(data=fd, plot_kws=dict(alpha=0.3))
g.fig.subplots_adjust(top=0.93)
g.fig.suptitle("Generated sample: G(Z)", fontsize=40)
for i in range(5):
    for j in range(4):
        g.axes[i, j].set_xlim(-3.5, 3.5)
        g.axes[j, i].set_ylim(-3.5, 3.5)
    # The last row
    g.axes[4, i].set_ylim(-6, 10)
    # The last column
    g.axes[i, 4].set_xlim(-6, 10)
# plt.show()
# Figure 2(c) in the main article
plt.savefig("./plots/hyperplane_gen.png", bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(6, 6))
sns.set(style="white", font_scale=1.5, palette="tab10")
g = sns.pairplot(data=recon, plot_kws=dict(alpha=0.3))
g.fig.subplots_adjust(top=0.93)
g.fig.suptitle("Reconstructed sample: G(Q(X))", fontsize=40)
for i in range(5):
    for j in range(4):
        g.axes[i, j].set_xlim(-3.5, 3.5)
        g.axes[j, i].set_ylim(-3.5, 3.5)
    # The last row
    g.axes[4, i].set_ylim(-6, 10)
    # The last column
    g.axes[i, 4].set_xlim(-6, 10)
# plt.show()
# Figure 2(c) in the main article
plt.savefig("./plots/hyperplane_recon.png", bbox_inches="tight")
plt.close()
