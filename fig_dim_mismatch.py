##### Swiss roll #####
import math
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.tools import set_random_seed, inf_train_gen, plot3d_gen
from models.blocks import gen_noise
from models.toy import Q_Toy, G_Toy, D_Toy
from models.wgan import WGAN
from models.wae import WAE

# True data
set_random_seed(123)
z_dim = 5
x_dim = 3
bsize = 2000
data = inf_train_gen("scurve", batch_size=bsize, std=0.0)
rd = next(data)
# Plot points
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.scatter(rd[:, 0], rd[:, 1], rd[:, 2], c=rd[:, 2], cmap=plt.cm.Spectral)
ax.view_init(15, -110)
ax.set_title("True Sample: X", fontsize=20, y=0.95)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.1, 2.1)
ax.set_zlim(-2.1, 2.1)
# plt.show()
# Figure 1(a) in the main article
plt.savefig("./plots/fig1_real.pdf", bbox_inches="tight")
plt.close()



# WGAN
set_random_seed(123)
use_cuda = torch.cuda.is_available()
dev = torch.device("cuda") if use_cuda else torch.device("cpu")

# Hyperparameters
x_dim = 3
z_dim = 1
g_dim = 64
d_dim = 64
nepoch = 20000
bsize = 512
lr = 1e-4
weight_decay = 0.0
iter_gq = 1
iter_d = 20
lambda_gp = 5.0
scheduler = 1
eval_freq = 100

# Initialize models
netG = G_Toy(z_dim=z_dim, struct_dim=g_dim, out_dim=x_dim).to(device=dev)
netD = D_Toy(input_dim=x_dim, struct_dim=d_dim).to(device=dev)
net = WGAN(z_dim, netG, netD, device=dev)
net = torch.jit.script(net)
netG, netD = net.netG, net.netD

# Load datasets
data = inf_train_gen("scurve", batch_size=bsize, std=0.0)

# Set up optimizers
optim_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
optim_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

# Set up schedulers
milestones = [int(math.ceil(nepoch * s)) for s in [0.8, 0.85, 0.9, 0.95]]
scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.9)
scheduler_D = optim.lr_scheduler.MultiStepLR(optim_D, milestones=milestones, gamma=0.9)

# Initial empty lists for training progress
RESULTS = {
    "epoch": [],
    "pre_critic_loss": [],
    "pre_grad_penalty": [],
    "post_critic_loss": [],
    "post_grad_penalty": []
}

# Compute loss function values
def eval_loss(x1, x2, z_dim):
    with torch.no_grad():
        critic_loss = -net.critic_diff(x1)
        n = x2.shape[0]
        z = gen_noise(n, z_dim, device=x2.device)
    grad_penalty = net.gradient_penalty_D(x2.data, z.data)
    return float(critic_loss), float(grad_penalty)

# Training
for epoch in tqdm(range(nepoch)):
    # 1. Update D network
    # (1). Set up parameters of D to update
    #      Freeze parameters of G
    for p in netD.parameters():
        p.requires_grad = True
    for p in netG.parameters():
        p.requires_grad = False

    # (2). Update D
    niter = 100 if epoch < 100 or epoch % 100 == 0 else iter_d
    for _ in range(niter):
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        netD.zero_grad()
        # D loss
        D_loss = net.D_loss(x1, x2, lambda_gp)
        D_loss.backward()
        optim_D.step()

    # Evaluate loss
    if epoch % eval_freq == 0:
        netG.eval()
        netD.eval()
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        pre_critic_loss, pre_grad_penalty = eval_loss(x1, x2, z_dim)
        RESULTS["epoch"].append(epoch)
        RESULTS["pre_critic_loss"].append(pre_critic_loss)
        RESULTS["pre_grad_penalty"].append(pre_grad_penalty)
        netG.train()
        netD.train()

    # 2. Update G network
    # (1). Set up parameters of G to update
    #      Freeze parameters of D
    for p in netD.parameters():
        p.requires_grad = False
    for p in netG.parameters():
        p.requires_grad = True

    # (2). Update G
    niter = iter_gq
    for _ in range(niter):
        x = torch.tensor(next(data), device=dev)
        netG.zero_grad()
        # G loss
        G_loss = net(x)
        G_loss.backward()
        optim_G.step()

    # End of epoch, call schedulers
    if scheduler > 0:
        scheduler_G.step()
        scheduler_D.step()

    # Evaluate loss
    if epoch % eval_freq == 0:
        netG.eval()
        netD.eval()
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        post_critic_loss, post_grad_penalty = eval_loss(x1, x2, z_dim)
        RESULTS["post_critic_loss"].append(post_critic_loss)
        RESULTS["post_grad_penalty"].append(post_grad_penalty)
        netG.train()
        netD.train()

# Plot losses
fig = plt.figure(figsize=(12, 10))
sub = fig.add_subplot(221)
plt.plot(RESULTS["pre_critic_loss"])
plt.title("Pre Critic Loss")
sub = fig.add_subplot(222)
plt.plot(RESULTS["pre_grad_penalty"])
plt.title("Pre Gradient Penalty")
sub = fig.add_subplot(223)
plt.plot(RESULTS["post_critic_loss"])
plt.title("Post Critic Loss")
sub = fig.add_subplot(224)
plt.plot(RESULTS["pre_grad_penalty"])
plt.title("Post Gradient Penalty")
plt.show()

# Plot generated points
# Figure 1(b) in the main article
plot3d_gen(netG, nsamp=2000, zdim=z_dim, rank=z_dim, device=dev,
           filename="./plots/fig1_wgan_gen.pdf")



# WAE
set_random_seed(123)
use_cuda = torch.cuda.is_available()
dev = torch.device("cuda") if use_cuda else torch.device("cpu")

# Hyperparameters
x_dim = 3
z_dim = 3
q_dim = 64
g_dim = 64
nepoch = 20000
bsize = 512
lr = 5e-4
weight_decay = 0.0
lambda_mmd = 100.0
scheduler = 1
eval_freq = 100

# Initialize models
netQ = Q_Toy(z_dim=z_dim, input_dim=x_dim, struct_dim=q_dim).to(device=dev)
netG = G_Toy(z_dim=z_dim, struct_dim=g_dim, out_dim=x_dim).to(device=dev)
net = WAE(z_dim, netQ, netG, device=dev)
net = torch.jit.script(net)
netQ, netG = net.netQ, net.netG

# Load datasets
data = inf_train_gen("scurve", batch_size=bsize, std=0.0)

# Set up optimizers
optim_Q = optim.Adam(netQ.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
optim_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

# Set up schedulers
milestones = [int(math.ceil(nepoch * s)) for s in [0.8, 0.85, 0.9, 0.95]]
scheduler_Q = optim.lr_scheduler.MultiStepLR(optim_Q, milestones=milestones, gamma=0.9)
scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.9)

# Initial empty lists for training progress
RESULTS = {
    "epoch": [],
    "recon_error": [],
    "mmd": []
}

# Compute loss function values
def eval_loss(x1, x2, lambda_mmd):
    with torch.no_grad():
        recon = net.recon_loss(x1)
        mmd = net.mmd_penalty(x2, lambda_mmd)
    return float(recon), float(mmd)

# Training
for epoch in tqdm(range(nepoch)):
    x1 = torch.tensor(next(data), device=dev)
    x2 = torch.tensor(next(data), device=dev)
    netQ.zero_grad()
    netG.zero_grad()
    # Loss
    loss = net(x1, x2, lambda_mmd)
    loss.backward()
    optim_Q.step()
    optim_G.step()

    # End of epoch, call schedulers
    if scheduler > 0:
        scheduler_Q.step()
        scheduler_G.step()

    # Evaluate loss
    if epoch % eval_freq == 0:
        netQ.eval()
        netG.eval()
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        recon_error, mmd = eval_loss(x1, x2, lambda_mmd)
        RESULTS["epoch"].append(epoch)
        RESULTS["recon_error"].append(recon_error)
        RESULTS["mmd"].append(mmd)
        netQ.train()
        netG.train()

# Plot losses
fig = plt.figure(figsize=(12, 6))
sub = fig.add_subplot(121)
plt.plot(RESULTS["recon_error"])
plt.title("Reconstruction Error")
sub = fig.add_subplot(122)
plt.plot(RESULTS["mmd"])
plt.title("MMD")
plt.show()

# Plot generated points
# Figure 1(c) in the main article
plot3d_gen(netG, nsamp=2000, zdim=z_dim, rank=z_dim, device=dev,
           filename="./plots/fig1_wae_gen.pdf")

# Latent space
data = inf_train_gen("scurve", batch_size=2000, std=0.0)
x = torch.tensor(next(data), device=dev)
with torch.no_grad():
    z_hat = netQ(x)
z_hat = pd.DataFrame(z_hat.detach().cpu().numpy(), columns=[f"Z{i}" for i in range(1, z_dim + 1)])

# fig = plt.figure(figsize=(8, 8))
# sns.pairplot(data=z_hat)
# plt.show()

fig = plt.figure(figsize=(6, 6))
sns.set(style="white", font_scale=1.5, palette="tab10")
g = sns.pairplot(data=z_hat, plot_kws=dict(s=10, alpha=0.15))
for i in range(3):
    for j in range(3):
        g.axes[i, j].set_xlim(-3.5, 3.5)
        g.axes[j, i].set_ylim(-3.5, 3.5)
# plt.show()
# Figure 1(d) in the main article
plt.savefig("./plots/fig1_wae_latent.pdf", bbox_inches="tight")
plt.close()
