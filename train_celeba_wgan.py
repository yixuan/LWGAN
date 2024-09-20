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
from models.celeba import G_CelebA, D_CelebA
from models.wgan import WGAN

parser = argparse.ArgumentParser(description="CelebA WGAN Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="CelebA", help="CelebA is trained.")
parser.add_argument("--data_dir", type=str, default="data/", help="The folder containing the face64.pt file.")
parser.add_argument("--random_seed", type=int, default=2024, help="Random seed for reproducibility.")
parser.add_argument("--z_dim_min", type=int, default=16, help="The minimum latent dimension. Not used in WGAN.")
parser.add_argument("--z_dim_max", type=int, default=48, help="The maximum latent dimension.")
parser.add_argument("--structure_dim", type=int, default=128, help="Structural dimension of the networks.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
parser.add_argument("--batch_size_eval", type=int, default=256, help="Batch size for evaluation.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs for WGAN training.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for all optimizers.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for all optimizers.")
parser.add_argument("--scheduler", type=int, default=1, help="Schedulers for the optimizers.")
parser.add_argument("--iter_gq", type=int, default=10, help="Number of encoder/generator updates on each batch.")
parser.add_argument("--iter_d", type=int, default=10, help="Number of critic updates on each batch.")
parser.add_argument("--lambda_mmd", type=float, default=0.0, help="Regularization parameter for the MMD term. Not used in WGAN.")
parser.add_argument("--lambda_gp", type=float, default=5.0, help="Regularization parameter for the gradient penalty term.")
parser.add_argument("--lambda_rank", type=float, default=0.01, help="Regularization parameter for the rank term. Not used in WGAN.")
parser.add_argument("--device", type=str, default="cpu", help="Device used for training.")

args = parser.parse_args()
############################################################
# For interactive running
# args = parser.parse_args([
#     "--data_dir", "./data/celeba/",
#     "--z_dim_min", "1", "--z_dim_max", "128",
#     "--structure_dim", "64",
#     "--batch_size", "128", "--batch_size_eval", "512",
#     "--epochs", "100000", "--learning_rate", "2e-4", "--weight_decay", "0.0",
#     "--scheduler", "1",
#     "--iter_gq", "1", "--iter_d", "10",
#     "--lambda_gp", "20.0",
#     "--device", "cuda"])
############################################################

# Path to output directory
FN = filename_celeba(args)
DIR = os.path.join("./outputs/WGAN/", args.dataset, FN)
if not os.path.exists(DIR):
    os.makedirs(DIR, exist_ok=True)

# Set random seed and computing device
set_random_seed(args.random_seed)
use_cuda = torch.cuda.is_available() and "cuda" in args.device
dev = torch.device(args.device) if use_cuda else torch.device("cpu")

# Initialize models
netG = G_CelebA(z_dim=args.z_dim_max, struct_dim=args.structure_dim, act="relu").to(device=dev)
netD = D_CelebA(z_dim=args.z_dim_max, struct_dim=args.structure_dim // 2, act="relu").to(device=dev)
net = WGAN(args.z_dim_max, netG, netD, device=dev)
net = torch.jit.script(net)
netG, netD = net.netG, net.netD

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
    "pre_critic_loss": [],
    "pre_grad_penalty": [],
    "post_critic_loss": [],
    "post_grad_penalty": []
}

# ***********************
# *** WGAN Algorithm ***
# ***********************
# Training
max_rank = args.z_dim_max
nepoch = args.epochs
eval_freq = 100

# Set up optimizers
optim_G = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
optim_D = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
if args.scheduler == 1:
    milestones = [int(math.ceil(nepoch * s)) for s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    scheduler_G = optim.lr_scheduler.MultiStepLR(optim_G, milestones=milestones, gamma=0.9)
    scheduler_D = optim.lr_scheduler.MultiStepLR(optim_D, milestones=milestones, gamma=0.9)
elif args.scheduler == 2:
    m1 = int(math.ceil(nepoch * 0.3))
    m2 = int(math.ceil(nepoch * 0.7))
    scheduler_G1 = optim.lr_scheduler.ConstantLR(optim_G, factor=1.0, total_iters=m1)
    scheduler_G2 = optim.lr_scheduler.LinearLR(optim_G, start_factor=1.0, end_factor=0.1, total_iters=m2)
    scheduler_G = optim.lr_scheduler.SequentialLR(
        optim_G, [scheduler_G1, scheduler_G2], milestones=[m1])
    scheduler_D1 = optim.lr_scheduler.ConstantLR(optim_D, factor=1.0, total_iters=m1)
    scheduler_D2 = optim.lr_scheduler.LinearLR(optim_D, start_factor=1.0, end_factor=0.1, total_iters=m2)
    scheduler_D = optim.lr_scheduler.SequentialLR(
        optim_D, [scheduler_D1, scheduler_D2], milestones=[m1])
elif args.scheduler == 3:
    scheduler_G = optim.lr_scheduler.OneCycleLR(optim_G, args.learning_rate, total_steps=nepoch)
    scheduler_D = optim.lr_scheduler.OneCycleLR(optim_D, args.learning_rate, total_steps=nepoch)

# s = torch.tensor([1.0], requires_grad=True)
# optim_G = optim.Adam([s], 1.0)
# sche1 = optim.lr_scheduler.ConstantLR(optim_G, factor=1.0, total_iters=20)
# sche2 = optim.lr_scheduler.LinearLR(optim_G, start_factor=1.0, end_factor=0.1, total_iters=10)
# sche = optim.lr_scheduler.SequentialLR(optim_G, [sche1, sche2], milestones=[20])
# for _ in range(100):
#     print(sche.get_last_lr())
#     optim_G.step()
#     sche.step()

def eval_loss(x1, x2, z_dim):
    with torch.no_grad():
        critic_loss = -net.critic_diff(x1)
        n = x2.shape[0]
        z = gen_noise(n, z_dim, device=x2.device)
    grad_penalty = net.gradient_penalty_D(x2.data, z.data)
    return float(critic_loss), float(grad_penalty)

for epoch in tqdm(range(nepoch)):
    # 1. Update D network
    # (1). Set up parameters of D to update
    #      Freeze parameters of G
    for p in netD.parameters():
        p.requires_grad = True
    for p in netG.parameters():
        p.requires_grad = False

    # (2). Update D
    niterd = args.iter_d
    # niter = math.ceil((niterd - args.iter_gq) * (1.0 - epoch / nepoch)) + args.iter_gq
    # niter = niterd
    niter = 100 if epoch < 25 or epoch % 500 == 0 else niterd
    for _ in range(niter):
        x1 = get_x(train_loader, data)
        x2 = get_x(train_loader, data)
        netD.zero_grad()
        # D loss
        D_loss = net.D_loss(x1, x2, args.lambda_gp)
        D_loss.backward()
        optim_D.step()

    # Evaluate loss
    if epoch % eval_freq == 0:
        netG.eval()
        netD.eval()
        x1 = get_x(test_loader, test_data)
        x2 = get_x(test_loader, test_data)
        pre_critic_loss, pre_grad_penalty = eval_loss(x1, x2, max_rank)
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
    niter = args.iter_gq
    for _ in range(niter):
        x = get_x(train_loader, data)
        netG.zero_grad()
        # G loss
        G_loss = net(x)
        G_loss.backward()
        optim_G.step()

    # End of epoch, call schedulers
    if args.scheduler > 0:
        scheduler_G.step()
        scheduler_D.step()

    # Evaluate loss
    if epoch % eval_freq == 0:
        netG.eval()
        netD.eval()
        x1 = get_x(test_loader, test_data)
        x2 = get_x(test_loader, test_data)
        post_critic_loss, post_grad_penalty = eval_loss(x1, x2, max_rank)
        RESULTS["post_critic_loss"].append(post_critic_loss)
        RESULTS["post_grad_penalty"].append(post_grad_penalty)
        netG.train()
        netD.train()

    # Save generated images
    if (epoch + 1) % 1000 == 0:
        netG.eval()
        noise = gen_noise(64, max_rank, device=dev)
        fake_data = 0.5 * (1.0 + netG(noise))
        torchvision.utils.save_image(fake_data, DIR + f"/fake_epoch{epoch + 1}.png", nrow=8)
        netG.train()

# Save model
torch.save(netG.state_dict(), DIR + "/netG.pt")
torch.save(netD.state_dict(), DIR + "/netD.pt")
torch.save(optim_G.state_dict(), DIR + "/optG.pt")
torch.save(optim_D.state_dict(), DIR + "/optD.pt")
with open(DIR + "/losses.json", mode="w") as f:
    json.dump(RESULTS, f, indent=2)

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
fig.savefig(DIR + "/losses.png")
plt.close()

# Plot generated data
set_random_seed(args.random_seed)
noise_src = torch.randn(64, max_rank, device=dev)
netG.eval()
fake_data = 0.5 * (1.0 + netG(noise_src))
torchvision.utils.save_image(fake_data, DIR + f"/fake.png", nrow=8)
