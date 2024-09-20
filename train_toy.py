import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# Notes on reproducibility
#
# In strict reproducibility mode, PyTorch will use deterministic algorithms for
# all operations, which may be very slow
#
# strict_reproducibility=False is generally sufficient to preserve reproducibility,
# as the random seed will still be used
strict_reproducibility = False
if strict_reproducibility:
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import torch
import torch.optim as optim
if strict_reproducibility:
    torch.use_deterministic_algorithms(True)

from utils.tools import *
from models.blocks import gen_noise_with_rank, mask_noise
from models.toy import Qrank_Toy, G_Toy, Drank_Toy
from models.lwgan import LWGAN

# Command line arguments
parser = argparse.ArgumentParser(description="Toy Examples of LWGAN Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="scurve", help="Choose among 3 datasets: swissroll, scurve, hyperplane.")
parser.add_argument("--random_seed", type=int, default=2024, help="Random seed for reproducibility.")
parser.add_argument("--x_dim", type=int, default=2, help="The ambient dimension of the data.")
parser.add_argument("--z_dim", type=int, default=5, help="The maximum latent dimension.")
parser.add_argument("--q_dim", type=int, default=64, help="Structural dimension of the encoder.")
parser.add_argument("--g_dim", type=int, default=64, help="Structural dimension of the generator.")
parser.add_argument("--d_dim", type=int, default=64, help="Structural dimension of the critic.")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
parser.add_argument("--batch_size_eval", type=int, default=512, help="Batch size for evaluation.")
parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs for LWGAN training.")
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
# common = ["--q_dim", "64", "--g_dim", "64", "--d_dim", "64",
#           "--learning_rate", "2e-4", "--scheduler", "1",
#           "--lambda_mmd", "1.0", "--lambda_rank", "0.01",
#           "--device", "cuda"]
# args = parser.parse_args(
#     ["--dataset", "scurve", "--x_dim", "3", "--z_dim", "5"] +
#     ["--epochs", "10000", "--iter_gq", "1", "--iter_d", "20"] + common)
# args = parser.parse_args(
#     ["--dataset", "swissroll", "--x_dim", "2", "--z_dim", "5"] +
#     ["--epochs", "5000", "--iter_gq", "5", "--iter_d", "20"] + common)
# args = parser.parse_args(
#     ["--dataset", "hyperplane", "--x_dim", "5", "--z_dim", "7"] +
#     ["--epochs", "10000", "--iter_gq", "1", "--iter_d", "20"] + common)
############################################################

# Path to output directory
FN = filename_toy(args)
DIR = os.path.join("./outputs/", args.dataset, FN)
print(DIR)
if not os.path.exists(DIR):
    os.makedirs(DIR, exist_ok=True)

# Set random seed and computing device
set_random_seed(args.random_seed)
use_cuda = torch.cuda.is_available() and "cuda" in args.device
dev = torch.device(args.device) if use_cuda else torch.device("cpu")

# Initialize models
netQ = Qrank_Toy(z_dim=args.z_dim, rank_min=1, input_dim=args.x_dim, struct_dim=args.q_dim).to(device=dev)
netG = G_Toy(z_dim=args.z_dim, struct_dim=args.g_dim, out_dim=args.x_dim).to(device=dev)
netD = Drank_Toy(z_dim=args.z_dim, rank_min=1, input_dim=args.x_dim, struct_dim=args.d_dim).to(device=dev)
net = LWGAN(args.z_dim, netQ, netG, netD, device=dev)
net = torch.jit.script(net)
netQ, netG, netD = net.netQ, net.netG, net.netD

# Load datasets
data = inf_train_gen(args.dataset, batch_size=args.batch_size, std=0.0)
data_eval = inf_train_gen(args.dataset, batch_size=args.batch_size_eval, std=0.0)

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

# Visualization parameters
xlim, ylim = None, None
if args.dataset == "swissroll":
    xlim, ylim = (-2.5, 3), (-2.5, 3)

# Training parameters
max_rank = args.z_dim
nepoch = args.epochs
loss_eval_freq = 10

# Set up optimizers
opt_args = dict(lr=args.learning_rate, betas=(0.5, 0.999), weight_decay=args.weight_decay)
optim_Q = optim.Adam(netQ.parameters(), **opt_args)
optim_G = optim.Adam(netG.parameters(), **opt_args)
optim_D = optim.Adam(netD.parameters(), **opt_args)

# Schedulers
if args.scheduler == 1:
    milestones = [int(math.ceil(nepoch * s)) for s in [0.8, 0.85, 0.9, 0.95]]
    scheduler_Q = optim.lr_scheduler.MultiStepLR(optim_Q, milestones=milestones, gamma=0.9)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.9)
    scheduler_D = optim.lr_scheduler.MultiStepLR(optim_D, milestones=milestones, gamma=0.9)
elif args.scheduler == 2:
    milestones = [int(math.ceil(nepoch * s)) for s in [0.2, 0.5]]
    scheduler_Q = optim.lr_scheduler.MultiStepLR(optim_Q, milestones=milestones, gamma=0.1)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.1)
    scheduler_D = optim.lr_scheduler.MultiStepLR(optim_D, milestones=milestones, gamma=0.1)
elif args.scheduler == 3:
    scheduler_Q = optim.lr_scheduler.OneCycleLR(optim_Q, max_lr=args.learning_rate, total_steps=nepoch, cycle_momentum=False)
    scheduler_G = optim.lr_scheduler.OneCycleLR(optim_G, max_lr=args.learning_rate, total_steps=nepoch, cycle_momentum=False)
    scheduler_D = optim.lr_scheduler.OneCycleLR(optim_D, max_lr=args.learning_rate, total_steps=nepoch, cycle_momentum=False)

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
def eval_loss_post(x1, x2, max_rank, rank, lambda_mmd):
    n = x1.shape[0]
    with torch.no_grad():
        noise = gen_noise_with_rank(n, max_rank, rank, device=x1.device)
        fake_data = netG(noise)
        post_critic_loss = -net.D_loss(x1, fake_data, rank, abs=False)
        post_recon_error = net.recon_loss(x1, rank)
        z = gen_noise_with_rank(n, max_rank, rank, device=x1.device)
        post_mmd = net.mmd_penalty(x2, rank, lambda_mmd)
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

# Update the encoder and generator
def update_GQ(x1, x2, rank: int, lambda_mmd: float, lambda_rank: float):
    netG.zero_grad()
    netQ.zero_grad()
    # Primal loss
    primal_cost = net(x1, x2, rank, lambda_mmd, lambda_rank)
    primal_cost.backward()
    optim_G.step()
    optim_Q.step()

# Plot generated data on each rank
def plot_gen_ranks(max_rank, fn_prefix):
    for i in range(max_rank):
        rank = i + 1
        filename = f"{fn_prefix}_rank{rank}.png"

        if args.x_dim == 2:
            plot2d_gen(netG, args.batch_size, max_rank, rank, dev, xlim, ylim, filename)
        elif args.x_dim == 3:
            plot3d_gen(netG, args.batch_size, max_rank, rank, dev, filename)
        else:
            plotrd_gen(netG, args.batch_size, max_rank, rank, dev, filename)

# ***********************
# *** LWGAN Algorithm ***
# ***********************
# Training
for epoch in tqdm(range(nepoch)):
    # Randomly select a rank for Z
    rank = torch.randint(1, max_rank + 1, (1,), device=dev)
    rank = int(rank)

    # 1. Update D network
    # (1). Set up parameters of D to update
    #      Freeze parameters of G and Q
    mode_D()

    # (2). Update D
    # Train D to near-optimality at the beginning and at every 100 iterations
    niter = 100 if epoch < 100 or epoch % 100 == 0 else args.iter_d
    for _ in range(niter):
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        update_D(x1, x2, rank, args.lambda_gp)

    # Evaluate loss
    if epoch % loss_eval_freq == 0:
        x1 = torch.tensor(next(data_eval), device=dev)
        x2 = torch.tensor(next(data_eval), device=dev)
        pre_critic_loss, pre_grad_penalty = eval_loss_pre(x1, x2, max_rank, rank)
        LOSSES["epoch"].append(epoch)
        LOSSES["pre_critic_loss"].append(pre_critic_loss)
        LOSSES["pre_grad_penalty"].append(pre_grad_penalty)

    # 2. Update G and Q networks
    # (1). Set up parameters of G and Q to update
    #      Freeze parameters of D
    mode_GQ()

    # (2). Update G and Q
    niter = args.iter_gq
    lmmd = 0.0 if epoch < nepoch // 2 else args.lambda_mmd
    for _ in range(niter):
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        update_GQ(x1, x2, rank, lmmd, args.lambda_rank)

    # Evaluate loss
    if epoch % loss_eval_freq == 0:
        x1 = torch.tensor(next(data_eval), device=dev)
        x2 = torch.tensor(next(data_eval), device=dev)
        post_critic_loss, post_grad_penalty, post_recon_error, post_mmd = \
            eval_loss_post(x1, x2, max_rank, rank, lmmd)
        LOSSES["post_critic_loss"].append(post_critic_loss)
        LOSSES["post_grad_penalty"].append(post_grad_penalty)
        LOSSES["post_recon_error"].append(post_recon_error)
        LOSSES["post_mmd"].append(post_mmd)

    # End of epoch, call schedulers
    if args.scheduler > 0:
        scheduler_Q.step()
        scheduler_G.step()
        scheduler_D.step()

    # Plot generated data every 1000 epochs
    if epoch % 1000 == 0:
        fn_prefix = DIR + f"/epoch{epoch}"
        plot_gen_ranks(max_rank, fn_prefix)

# Save models
torch.save(netQ.state_dict(), DIR + "/netQ.pt")
torch.save(netG.state_dict(), DIR + "/netG.pt")
torch.save(netD.state_dict(), DIR + "/netD.pt")
torch.save(optim_Q.state_dict(), DIR + "/optQ.pt")
torch.save(optim_G.state_dict(), DIR + "/optG.pt")
torch.save(optim_D.state_dict(), DIR + "/optD.pt")
with open(DIR + "/losses.json", mode="w") as f:
    json.dump(LOSSES, f, indent=2)

# Load model
# netQ.load_state_dict(torch.load(DIR + "/netQ.pt", weights_only=True))
# netG.load_state_dict(torch.load(DIR + "/netG.pt", weights_only=True))
# netD.load_state_dict(torch.load(DIR + "/netD.pt", weights_only=True))
# optim_Q.load_state_dict(torch.load(DIR + "/optQ.pt", weights_only=True))
# optim_G.load_state_dict(torch.load(DIR + "/optG.pt", weights_only=True))
# optim_D.load_state_dict(torch.load(DIR + "/optD.pt", weights_only=True))
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
fn_prefix = DIR + "/generated"
plot_gen_ranks(max_rank, fn_prefix)

# Post-training evaluation of the rank scores
set_random_seed(args.random_seed)
nrep = 50
score_eval_freq = 20

# Statistics to estimate lambda
STATS = []

for i in tqdm(range(nrep * score_eval_freq)):
    # Randomly select a rank for Z
    rank = torch.randint(1, max_rank + 1, (1,), device=dev)
    rank = int(rank)

    # 1. Update D network
    # (1). Set up parameters of D to update
    #      Freeze parameters of G and Q
    mode_D()

    # (2). Update D
    niter = max(20, args.iter_d)
    for _ in range(niter):
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        update_D(x1, x2, rank, args.lambda_gp)

    # 2. Update G and Q networks
    # (1). Set up parameters of G and Q to update
    #      Freeze parameters of D
    mode_GQ()

    # (2). Update G and Q
    niter = args.iter_gq
    lmmd = args.lambda_mmd
    for _ in range(niter):
        x1 = torch.tensor(next(data), device=dev)
        x2 = torch.tensor(next(data), device=dev)
        update_GQ(x1, x2, rank, lmmd, args.lambda_rank)

    # Compute rank-wise statistics every score_eval_freq iterations
    if i % score_eval_freq == 0:
        with torch.no_grad():
            src1 = torch.randn(args.batch_size_eval, max_rank, device=dev)
            src2 = torch.randn(args.batch_size_eval, max_rank, device=dev)
            real_data = torch.tensor(next(data_eval), device=dev)
            x = torch.tensor(next(data_eval), device=dev)

        dat = []
        for rank in range(1, max_rank + 1):
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
# summ = stats.groupby(["rank"])[stats.keys().drop(["rep", "rank"])].mean().reset_index()
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