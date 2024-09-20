import argparse
import os
import json
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils.tools import *
from models.blocks import gen_noise
from models.celeba import Q_CelebA, G_CelebA
from models.wae import WAE

parser = argparse.ArgumentParser(description="CelebA WAE Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="CelebA", help="CelebA is trained.")
parser.add_argument("--data_dir", type=str, default="data/", help="The folder containing the face64.pt file.")
parser.add_argument("--random_seed", type=int, default=2024, help="Random seed for reproducibility.")
parser.add_argument("--z_dim_min", type=int, default=16, help="The minimum latent dimension. Not used in WAE.")
parser.add_argument("--z_dim_max", type=int, default=48, help="The maximum latent dimension.")
parser.add_argument("--structure_dim", type=int, default=128, help="Structural dimension of the networks.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
parser.add_argument("--batch_size_eval", type=int, default=256, help="Batch size for evaluation.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs for WAE training.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for all optimizers.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for all optimizers.")
parser.add_argument("--scheduler", type=int, default=1, help="Schedulers for the optimizers.")
parser.add_argument("--iter_gq", type=int, default=10, help="Number of encoder/generator updates on each batch. Not used in WAE.")
parser.add_argument("--iter_d", type=int, default=10, help="Number of critic updates on each batch. Not used in WAE.")
parser.add_argument("--lambda_mmd", type=float, default=10.0, help="Regularization parameter for the MMD term.")
parser.add_argument("--lambda_gp", type=float, default=5.0, help="Regularization parameter for the gradient penalty term. Not used in WAE.")
parser.add_argument("--lambda_rank", type=float, default=0.01, help="Regularization parameter for the rank term. Not used in WAE.")
parser.add_argument("--device", type=str, default="cpu", help="Device used for training.")

args = parser.parse_args()
############################################################
# For interactive running
# args = parser.parse_args([
#     "--data_dir", "./data/celeba/",
#     "--z_dim_min", "1", "--z_dim_max", "128",
#     "--structure_dim", "64",
#     "--batch_size", "128", "--batch_size_eval", "512",
#     "--epochs", "100000", "--learning_rate", "5e-4", "--weight_decay", "0.0",
#     "--scheduler", "1",
#     "--iter_gq", "1", "--iter_d", "10",
#     "--lambda_mmd", "100.0",
#     "--device", "cuda"])
############################################################

# Path to output directory
FN = filename_celeba(args)
DIR = os.path.join("./outputs/WAE/", args.dataset, FN)
if not os.path.exists(DIR):
    os.makedirs(DIR, exist_ok=True)

# Set random seed and computing device
set_random_seed(args.random_seed)
use_cuda = torch.cuda.is_available() and "cuda" in args.device
dev = torch.device(args.device) if use_cuda else torch.device("cpu")

# Initialize models
netQ = Q_CelebA(z_dim=args.z_dim_max, struct_dim=args.structure_dim, act="relu").to(device=dev)
netG = G_CelebA(z_dim=args.z_dim_max, struct_dim=args.structure_dim, act="relu").to(device=dev)
net = WAE(args.z_dim_max, netQ, netG, device=dev)
net = torch.jit.script(net)
netQ, netG = net.netQ, net.netG

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
RESULTS = {
    "recon_error": [],
    "mmd": []
}

# ***********************
# *** WAE Algorithm ***
# ***********************
# Training
max_rank = args.z_dim_max
nepoch = args.epochs
eval_freq = 100

# Set up optimizers
optim_Q = optim.Adam(netQ.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
optim_G = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
if args.scheduler == 1:
    milestones = [int(math.ceil(nepoch * s)) for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
    scheduler_Q = optim.lr_scheduler.MultiStepLR(optim_Q, milestones=milestones, gamma=0.9)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.9)
elif args.scheduler == 2:
    milestones = [int(math.ceil(nepoch * s)) for s in [0.7, 0.8, 0.9]]
    scheduler_Q = optim.lr_scheduler.MultiStepLR(optim_Q, milestones=milestones, gamma=0.2)
    scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.2)
elif args.scheduler == 3:
    scheduler_Q = optim.lr_scheduler.OneCycleLR(optim_Q, args.learning_rate, total_steps=nepoch)
    scheduler_G = optim.lr_scheduler.OneCycleLR(optim_G, args.learning_rate, total_steps=nepoch)

def eval_loss(x1, x2, lambda_mmd):
    with torch.no_grad():
        recon = net.recon_loss(x1)
        mmd = net.mmd_penalty(x2, lambda_mmd)
    return float(recon), float(mmd)

for epoch in tqdm(range(nepoch)):
    x1 = get_x(train_loader, data)
    x2 = get_x(train_loader, data)
    netQ.zero_grad()
    netG.zero_grad()
    # Loss
    loss = net(x1, x2, args.lambda_mmd)
    loss.backward()
    optim_Q.step()
    optim_G.step()

    # End of epoch, call schedulers
    if args.scheduler > 0:
        scheduler_Q.step()
        scheduler_G.step()

    # Evaluate loss
    if epoch % eval_freq == 0:
        netQ.eval()
        netG.eval()
        x1 = get_x(test_loader, test_data)
        x2 = get_x(test_loader, test_data)
        recon_error, mmd = eval_loss(x1, x2, args.lambda_mmd)
        RESULTS["recon_error"].append(recon_error)
        RESULTS["mmd"].append(mmd)
        netQ.train()
        netG.train()

    # Save generated images
    if (epoch + 1) % 1000 == 0:
        netG.eval()
        noise = gen_noise(64, max_rank, device=dev)
        fake_data = 0.5 * (1.0 + netG(noise))
        torchvision.utils.save_image(fake_data, DIR + f"/fake_epoch{epoch + 1}.png", nrow=8)
        netG.train()

# Save model
torch.save(netQ.state_dict(), DIR + "/netQ.pt")
torch.save(netG.state_dict(), DIR + "/netG.pt")
torch.save(optim_Q.state_dict(), DIR + "/optQ.pt")
torch.save(optim_G.state_dict(), DIR + "/optG.pt")
with open(DIR + "/losses.json", mode="w") as f:
    json.dump(RESULTS, f, indent=2)

# Plot losses
fig = plt.figure(figsize=(12, 6))
sub = fig.add_subplot(121)
plt.plot(RESULTS["recon_error"])
plt.title("Reconstruction Error")
sub = fig.add_subplot(122)
plt.plot(RESULTS["mmd"])
plt.title("MMD")
plt.show()
fig.savefig(DIR + "/losses.png")
plt.close()

# Plot generated data
set_random_seed(args.random_seed)
noise_src = torch.randn(64, max_rank, device=dev)
netG.eval()
fake_data = 0.5 * (1.0 + netG(noise_src))
torchvision.utils.save_image(fake_data, DIR + f"/fake.png", nrow=8)
