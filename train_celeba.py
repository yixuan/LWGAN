import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils.tools import *
from models.blocks import gen_noise_with_rank, mask_noise
from models.celeba import Qrank_CelebA, G_CelebA, Drank_CelebA
from models.lwgan import LWGAN

# Command line arguments
parser = argparse.ArgumentParser(description="CelebA LWGAN Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="CelebA", help="CelebA is trained.")
parser.add_argument("--data_dir", type=str, default="data/", help="The folder containing the face64.pt file.")
parser.add_argument("--random_seed", type=int, default=2024, help="Random seed for reproducibility.")
parser.add_argument("--z_dim_min", type=int, default=16, help="The minimum latent dimension.")
parser.add_argument("--z_dim_max", type=int, default=48, help="The maximum latent dimension.")
parser.add_argument("--structure_dim", type=int, default=128, help="Structural dimension of the encoder, generator, and critic.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
parser.add_argument("--batch_size_eval", type=int, default=256, help="Batch size for evaluation.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs for LWGAN training.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for all optimizers.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for all optimizers.")
parser.add_argument("--scheduler", type=int, default=1, help="Schedulers for the optimizers.")
parser.add_argument("--iter_gq", type=int, default=10, help="Number of encoder/generator updates on each batch.")
parser.add_argument("--iter_d", type=int, default=10, help="Number of critic updates on each batch.")
parser.add_argument("--lambda_mmd", type=float, default=0.0, help="Regularization parameter for the MMD term.")
parser.add_argument("--lambda_gp", type=float, default=5.0, help="Regularization parameter for the gradient penalty term.")
parser.add_argument("--lambda_rank", type=float, default=0.01, help="Regularization parameter for the rank term.")
parser.add_argument("--device", type=str, default="cpu", help="Device used for training.")

############################################################
# For command line running
args = parser.parse_args()
############################################################
# For interactive running
# args = parser.parse_args([
#     "--data_dir", "./data/celeba/",
#     "--z_dim_min", "1", "--z_dim_max", "128",
#     "--structure_dim", "64",
#     "--batch_size", "128", "--batch_size_eval", "512",
#     "--epochs", "100000", "--learning_rate", "1e-4", "--weight_decay", "0.0",
#     "--scheduler", "1",
#     "--iter_gq", "5", "--iter_d", "10",
#     "--lambda_mmd", "1.0", "--lambda_gp", "5.0", "--lambda_rank", "0.002",
#     "--device", "cuda"])
############################################################

# Path to output directory
FN = filename_celeba(args)
DIR = os.path.join("./outputs/", args.dataset, FN)
if not os.path.exists(DIR):
    os.makedirs(DIR, exist_ok=True)

# Set random seed and computing device
set_random_seed(args.random_seed)
use_cuda = torch.cuda.is_available() and "cuda" in args.device
dev = torch.device(args.device) if use_cuda else torch.device("cpu")

# Initialize models
netQ = Qrank_CelebA(z_dim=args.z_dim_max, rank_min=args.z_dim_min, struct_dim=args.structure_dim, act="relu").to(device=dev)
netG = G_CelebA(z_dim=args.z_dim_max, struct_dim=args.structure_dim, act="relu").to(device=dev)
netD = Drank_CelebA(z_dim=args.z_dim_max, rank_min=args.z_dim_min, struct_dim=args.structure_dim // 2, act="relu").to(device=dev)
net = LWGAN(args.z_dim_max, netQ, netG, netD, device=dev)
net = torch.jit.script(net)
netQ, netG, netD = net.netQ, net.netG, net.netD

# Load datasets
face64 = torch.load(
    os.path.join(args.data_dir, "face64.pt"),
    map_location=dev, weights_only=True)
train_loader = DataLoader(face64, batch_size=args.batch_size, shuffle=True, drop_last=True)
data = iter(train_loader)
test_loader = DataLoader(face64, batch_size=args.batch_size_eval, shuffle=True, drop_last=True)
test_data = iter(test_loader)
def get_x(loader, loader_iter):
    try:
        _data = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        _data = next(loader_iter)
    return _data

# Initial empty lists for training progress
LOSSES = {
    "epoch": [],
    "pre_critic_loss": [],
    "pre_grad_penalty": [],
    "post_critic_loss": [],
    "post_grad_penalty": [],
    "post_recon_error": [],
    "post_mmd": []
}

# Training parameters
max_rank = args.z_dim_max
min_rank = args.z_dim_min
nepoch = args.epochs
nepoch1 = (2 * nepoch) // 3
nepoch2 = nepoch - nepoch1
loss_eval_freq = 100

# Set up optimizers
lr = args.learning_rate
optim_Q = optim.Adam(netQ.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
optim_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
optim_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
# Second stage
optim2_Q = optim.SGD(netQ.parameters(), lr=lr, weight_decay=args.weight_decay)
optim2_G = optim.SGD(netG.parameters(), lr=lr, weight_decay=args.weight_decay)
optim2_D = optim.SGD(netD.parameters(), lr=lr, weight_decay=args.weight_decay)

# Schedulers
if args.scheduler == 1:
    # First stage
    milestones = [int(math.ceil(nepoch1 * s)) for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
    scheduler_Q = optim.lr_scheduler.MultiStepLR(optim_Q, milestones=milestones, gamma=0.9)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.9)
    scheduler_D = optim.lr_scheduler.MultiStepLR(optim_D, milestones=milestones, gamma=0.9)
    # Second stage
    optim2_Q = optim.SGD(netQ.parameters(), lr=lr * (0.9 ** 5))
    optim2_G = optim.SGD(netG.parameters(), lr=lr * (0.9 ** 5))
    optim2_D = optim.SGD(netD.parameters(), lr=lr * (0.9 ** 5), weight_decay=args.weight_decay)
    milestones = [int(math.ceil(nepoch2 * s)) for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
    scheduler2_Q = optim.lr_scheduler.MultiStepLR(optim2_Q, milestones=milestones, gamma=0.9)
    scheduler2_G = optim.lr_scheduler.MultiStepLR(optim2_G, milestones=milestones, gamma=0.9)
    scheduler2_D = optim.lr_scheduler.MultiStepLR(optim2_D, milestones=milestones, gamma=0.9)
elif args.scheduler == 2:
    scheduler_Q = optim.lr_scheduler.OneCycleLR(optim_Q, lr, total_steps=nepoch1)
    scheduler_G = optim.lr_scheduler.OneCycleLR(optim_G, lr, total_steps=nepoch1)
    scheduler_D = optim.lr_scheduler.OneCycleLR(optim_D, lr, total_steps=nepoch1)
    scheduler2_Q = optim.lr_scheduler.OneCycleLR(optim2_Q, lr, total_steps=nepoch2)
    scheduler2_G = optim.lr_scheduler.OneCycleLR(optim2_G, lr, total_steps=nepoch2)
    scheduler2_D = optim.lr_scheduler.OneCycleLR(optim2_D, lr, total_steps=nepoch2)

# Functions used in training and evaluation
# Evaluate losses before G and Q updates
def eval_loss_pre(x1, x2, max_rank, rank):
    n = x1.shape[0]
    with torch.no_grad():
        noise = gen_noise_with_rank(n, max_rank, rank, device=x1.device)
        fake_data = netG(noise)
        pre_critic_loss = -net.D_loss(x1, fake_data, rank, abs=False)
        z = gen_noise_with_rank(n, max_rank, rank, device=x1.device)
    pre_grad_penalty = net.gradient_penalty_D(x2.data, z.data, rank)
    return float(pre_critic_loss), float(pre_grad_penalty)

# Evaluate losses after G and Q updates
def eval_loss_post(x1, x2, max_rank, rank):
    n = x1.shape[0]
    with torch.no_grad():
        noise = gen_noise_with_rank(n, max_rank, rank, device=x1.device)
        fake_data = netG(noise)
        post_critic_loss = -net.D_loss(x1, fake_data, rank, abs=False)
        post_recon_error = net.recon_loss(x1, rank)
        z = gen_noise_with_rank(n, max_rank, rank, device=x1.device)
        post_mmd = net.mmd_penalty(x2, rank, args.lambda_mmd)
    post_grad_penalty = net.gradient_penalty_D(x2.data, z.data, rank)
    return float(post_critic_loss), float(post_grad_penalty), \
           float(post_recon_error), float(post_mmd)

# Set up parameters of D to update
# Freeze parameters of G and Q
def mode_D():
    for p in netD.parameters():
        p.requires_grad = True
    for p in netQ.parameters():
        p.requires_grad = False
    for p in netG.parameters():
        p.requires_grad = False
    netD.train()
    netG.train()
    netQ.train()

# Update the critic
def update_D(x1, x2, rank: int, lambda_gp: float):
    netD.zero_grad()
    # Dual loss
    dual_cost = net.dual_loss(x1, x2, rank, lambda_gp)
    dual_cost.backward()
    optim_D.step()

# Set up parameters of G and Q to update
# Freeze parameters of D
def mode_GQ():
    for p in netD.parameters():
        p.requires_grad = False
    for p in netQ.parameters():
        p.requires_grad = True
    for p in netG.parameters():
        p.requires_grad = True
    netD.train()
    netG.train()
    netQ.train()

# Update the encoder and generator
def update_GQ(x1, x2, rank: int, lambda_mmd: float, lambda_rank: float):
    netG.zero_grad()
    netQ.zero_grad()
    # Primal loss
    primal_cost = net(x1, x2, rank, lambda_mmd, lambda_rank)
    primal_cost.backward()
    optim_G.step()
    optim_Q.step()

# ***********************
# *** LWGAN Algorithm ***
# ***********************
# Training
for epoch in tqdm(range(nepoch)):
    # Randomly select a rank for Z
    rank = torch.randint(min_rank, max_rank + 1, (1,), device=dev)
    rank = int(rank)

    # Select optimizers according to stage
    if epoch >= nepoch1:
        optim_Q, optim_G, optim_D = optim2_Q, optim2_G, optim2_D
        if args.scheduler > 0:
            scheduler_Q, scheduler_G, scheduler_D = scheduler2_Q, scheduler2_G, scheduler2_D
    # For debugging purposes, print current learning rate
    if epoch % 1000 == 0 and args.scheduler > 0:
        lr_Q, lr_G, lr_D = scheduler_Q.get_last_lr(), scheduler_G.get_last_lr(), scheduler_D.get_last_lr()
        print(f"epoch = {epoch}, lr_Q = {lr_Q}, lr_G = {lr_G}, lr_D = {lr_D}")
        print(optim_Q)
        print(scheduler_Q)

    # 1. Update D network
    # (1). Set up parameters of D to update
    #      Freeze parameters of G and Q
    mode_D()

    # (2). Update D
    # Train D to near-optimality at the beginning and at every 100 iterations
    niter = 100 if epoch < 25 or epoch % 500 == 0 else args.iter_d
    for _ in range(niter):
        x1 = get_x(train_loader, data)
        x2 = get_x(train_loader, data)
        update_D(x1, x2, rank, args.lambda_gp)

    # Evaluate loss
    if epoch % loss_eval_freq == 0:
        netG.eval()
        netD.eval()
        x1 = get_x(test_loader, test_data)
        x2 = get_x(test_loader, test_data)
        pre_critic_loss, pre_grad_penalty = eval_loss_pre(x1, x2, max_rank, rank)
        LOSSES["epoch"].append(epoch)
        LOSSES["pre_critic_loss"].append(pre_critic_loss)
        LOSSES["pre_grad_penalty"].append(pre_grad_penalty)
        netG.train()
        netD.train()

    # 2. Update G and Q networks
    # (1). Set up parameters of G and Q to update
    #      Freeze parameters of D
    mode_GQ()

    # (2). Update G and Q
    niter = args.iter_gq
    for _ in range(niter):
        x1 = get_x(train_loader, data)
        x2 = get_x(train_loader, data)
        update_GQ(x1, x2, rank, args.lambda_mmd, args.lambda_rank)

    # Evaluate loss
    if epoch % loss_eval_freq == 0:
        netG.eval()
        netD.eval()
        x1 = get_x(test_loader, test_data)
        x2 = get_x(test_loader, test_data)
        post_critic_loss, post_grad_penalty, post_recon_error, post_mmd = \
            eval_loss_post(x1, x2, max_rank, rank)
        LOSSES["post_critic_loss"].append(post_critic_loss)
        LOSSES["post_grad_penalty"].append(post_grad_penalty)
        LOSSES["post_recon_error"].append(post_recon_error)
        LOSSES["post_mmd"].append(post_mmd)
        netG.train()
        netD.train()

    # End of epoch, call schedulers
    if args.scheduler > 0:
        scheduler_Q.step()
        scheduler_G.step()
        scheduler_D.step()

    # Save generated images
    if (epoch + 1) % 1000 == 0:
        netG.eval()
        noise = gen_noise_with_rank(64, max_rank, rank, device=dev)
        fake_data = 0.5 * (1.0 + netG(noise))
        torchvision.utils.save_image(fake_data, DIR + f"/fake_epoch{epoch + 1}_rank{int(rank)}.png", nrow=8)
        netG.train()

# Save model
torch.save(netQ.state_dict(), DIR + "/netQ.pt")
torch.save(netG.state_dict(), DIR + "/netG.pt")
torch.save(netD.state_dict(), DIR + "/netD.pt")
torch.save(optim_Q.state_dict(), DIR + "/optQ.pt")
torch.save(optim_G.state_dict(), DIR + "/optG.pt")
torch.save(optim_D.state_dict(), DIR + "/optD.pt")
torch.save(optim2_Q.state_dict(), DIR + "/opt2Q.pt")
torch.save(optim2_G.state_dict(), DIR + "/opt2G.pt")
torch.save(optim2_D.state_dict(), DIR + "/opt2D.pt")
with open(DIR + "/losses.json", mode="w") as f:
    json.dump(LOSSES, f, indent=2)

# Load model
# netQ.load_state_dict(torch.load(DIR + "/netQ.pt", weights_only=True))
# netG.load_state_dict(torch.load(DIR + "/netG.pt", weights_only=True))
# netD.load_state_dict(torch.load(DIR + "/netD.pt", weights_only=True))
# optim_Q.load_state_dict(torch.load(DIR + "/optQ.pt", weights_only=True))
# optim_G.load_state_dict(torch.load(DIR + "/optG.pt", weights_only=True))
# optim_D.load_state_dict(torch.load(DIR + "/optD.pt", weights_only=True))
# optim2_Q.load_state_dict(torch.load(DIR + "/opt2Q.pt", weights_only=True))
# optim2_G.load_state_dict(torch.load(DIR + "/opt2G.pt", weights_only=True))
# optim2_D.load_state_dict(torch.load(DIR + "/opt2D.pt", weights_only=True))
# with open(DIR + "/losses.json", mode="r") as f:
#     LOSSES = json.load(f)

# Plot losses
fig = plt.figure(figsize=(15, 10))
sub = fig.add_subplot(231)
plt.plot(LOSSES["pre_critic_loss"])
plt.title("Pre Critic Loss")
sub = fig.add_subplot(232)
plt.plot(LOSSES["pre_grad_penalty"])
plt.title("Pre Gradient Penalty")
sub = fig.add_subplot(234)
plt.plot(LOSSES["post_critic_loss"])
plt.title("Post Critic Loss")
sub = fig.add_subplot(235)
plt.plot(LOSSES["pre_grad_penalty"])
plt.title("Post Gradient Penalty")
sub = fig.add_subplot(233)
plt.plot(LOSSES["post_recon_error"])
plt.title("Post Reconstruction Error")
sub = fig.add_subplot(236)
plt.plot(LOSSES["post_mmd"])
plt.title("Post MMD")
plt.show()
fig.savefig(DIR + "/losses.png")
plt.close()

# Plot generated data for each rank
set_random_seed(args.random_seed)
noise_src = torch.randn(64, max_rank, device=dev)
netG.eval()
for rank in range(min_rank, max_rank + 1):
    noise = mask_noise(noise_src, rank)
    fake_data = 0.5 * (1.0 + netG(noise))
    torchvision.utils.save_image(fake_data, DIR + f"/fake_rank{rank}.png", nrow=8)

# Post-training evaluation of the rank scores
set_random_seed(args.random_seed)
nrep = 50
score_eval_freq = 20

# Statistics to estimate lambda
STATS = []

for i in tqdm(range(nrep * score_eval_freq)):
    # Randomly select a rank for Z
    rank = torch.randint(min_rank, max_rank + 1, (1,), device=dev)
    rank = int(rank)

    # 1. Update D network
    # (1). Set up parameters of D to update
    #      Freeze parameters of G and Q
    mode_D()

    # (2). Update D
    # Train D to near-optimality at the beginning and at every 100 iterations
    niter = max(20, args.iter_d)
    for _ in range(niter):
        x1 = get_x(train_loader, data)
        x2 = get_x(train_loader, data)
        update_D(x1, x2, rank, args.lambda_gp)

    # 2. Update G and Q networks
    # (1). Set up parameters of G and Q to update
    #      Freeze parameters of D
    mode_GQ()

    # (2). Update G and Q
    niter = args.iter_gq
    for _ in range(niter):
        x1 = get_x(train_loader, data)
        x2 = get_x(train_loader, data)
        update_GQ(x1, x2, rank, args.lambda_mmd, args.lambda_rank)

    # Compute rank-wise statistics every score_eval_freq iterations
    if i % score_eval_freq == 0:
        with torch.no_grad():
            src1 = torch.randn(args.batch_size_eval, max_rank, device=dev)
            src2 = torch.randn(args.batch_size_eval, max_rank, device=dev)
            real_data = get_x(test_loader, test_data)
            x = get_x(test_loader, test_data)

        dat = []
        for rank in range(min_rank, max_rank + 1):
            # Evaluate loss
            with torch.no_grad():
                noise = mask_noise(src1, rank)
                z = mask_noise(src2, rank)
                fake_data = netG(noise)
                critic_loss = -net.D_loss(real_data, fake_data, rank, abs=False)
                recon_error = net.recon_loss(real_data, rank)
                mmd = net.mmd_penalty(x, rank, args.lambda_mmd)
            grad_penalty = net.gradient_penalty_D(x.data, z.data, rank)
            dati = dict(
                rep=i // score_eval_freq + 1,
                rank=rank,
                critic_loss=float(critic_loss), grad_penalty=float(grad_penalty),
                recon_error=float(recon_error), mmd=float(mmd))
            dat.append(dati)
        dat = pd.DataFrame.from_records(dat)
        STATS.append(dat)

# Save statistics
stats = pd.concat(STATS, ignore_index=True)
stats.to_csv(DIR + "/stats.csv", index=False)

# Estimate lambda for rank
stats = pd.read_csv(DIR + "/stats.csv")
nrep = stats["rep"].values.max()
lst = []
for i in range(1, nrep + 1):
    dat = stats[stats["rep"] == i]
    loss = dat["critic_loss"] + dat["recon_error"] + dat["grad_penalty"] + dat["mmd"]
    loss = loss.values
    lst.append(loss)
mat = np.array(lst)
score_mean = np.mean(mat, axis=0)
score_se = np.std(mat, axis=0, ddof=1) / np.sqrt(nrep)
lam_rank = score_se[np.argmin(score_mean)] ** 0.8

# Stats across ranks
summ = stats.groupby(["rank"]).mean().drop(columns="rep").reset_index()
w0 = summ["critic_loss"] + summ["recon_error"]
w1 = w0 + summ["grad_penalty"] + summ["mmd"]
w2 = w1 + args.lambda_rank * summ["rank"]
w3 = w1 + lam_rank * summ["rank"]

fig = plt.figure(figsize=(15, 10))
sub = fig.add_subplot(241)
plt.plot(summ["rank"], summ["critic_loss"])
plt.title("Critic Loss")
plt.xlabel("Rank")
sub = fig.add_subplot(242)
plt.plot(summ["rank"], summ["recon_error"])
plt.title("Reconstruction Error")
plt.xlabel("Rank")
sub = fig.add_subplot(243)
plt.plot(summ["rank"], summ["grad_penalty"])
plt.title("Gradient Penalty")
plt.xlabel("Rank")
sub = fig.add_subplot(244)
plt.plot(summ["rank"], summ["mmd"])
plt.title("MMD")
plt.xlabel("Rank")
sub = fig.add_subplot(245)
plt.plot(summ["rank"], w0)
plt.title("W0 (Critic + Recon.)")
plt.xlabel("Rank")
sub = fig.add_subplot(246)
plt.plot(summ["rank"], w1)
plt.title("W1 (W0 + Grad. + MMD)")
plt.xlabel("Rank")
sub = fig.add_subplot(247)
plt.plot(summ["rank"], w2)
plt.title("W1 + Rank (arg)")
plt.xlabel("Rank")
sub = fig.add_subplot(248)
plt.plot(summ["rank"], w3)
plt.title("W1 + Rank (est)")
plt.xlabel("Rank")
plt.show()
fig.savefig(DIR + "/rank.png")
plt.close()

# Save rank information
with open(DIR + "/rank.json", mode="w") as f:
    ranks = summ["rank"].values.tolist()
    rank_est = ranks[np.argmin(w3)]
    res = dict(ranks=ranks, lam_est=float(lam_rank), rank_est=rank_est)
    json.dump(res, f, indent=2)
summ.to_csv(DIR + "/summ.csv", index=False)
