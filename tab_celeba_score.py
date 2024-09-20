# https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
# https://torchmetrics.readthedocs.io/en/stable/image/inception_score.html
# https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import PIL.Image as Image
import numpy as np
import torch
torch.use_deterministic_algorithms(True)
from torchvision.transforms import transforms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from models.blocks import mask_noise
from models.celeba import Qrank_CelebA, Q_CelebA, G_CelebA
from models.losses import mmd_penalty

# The path to the face64.pt file
path = "./data/celeba/face64.pt"

use_cuda = torch.cuda.is_available()
dev = torch.device("cuda") if use_cuda else torch.device("cpu")
z_dim = 128
bsize = 1000
nbatch = 5

def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.PILToTensor()(resized_img))
    return torch.stack(arr).to(device=dev)

def gen_data(G, z_dim, bsize, nbatch, rank=None):
    G.eval()
    fake_data = []
    for _ in range(nbatch):
        with torch.no_grad():
            noise = torch.randn(bsize, z_dim, device=dev)
            if rank is not None:
                noise = mask_noise(noise, rank)
            img = 0.5 * (1.0 + G(noise))
            fake_data.append(interpolate(img))
    torch.cuda.empty_cache()
    return fake_data

# Inception score
def inception_score(fake_data):
    inception = InceptionScore(normalize=False).to(device=dev)
    for fd in fake_data:
        inception.update(fd)
    mean, std = inception.compute()
    print(f"Inception score = {mean} ({std})")
    del inception
    torch.cuda.empty_cache()
    return mean, std

# Fréchet inception distance
def fid_score(fake_data, real_data):
    fid = FrechetInceptionDistance(normalize=False).to(device=dev)
    for rd, fd in zip(real_data, fake_data):
        with torch.no_grad():
            fid.update(rd, real=True)
            torch.cuda.empty_cache()
            fid.update(fd, real=False)
            torch.cuda.empty_cache()
    score = fid.compute()
    print(f"FID score = {score}")
    del fid
    torch.cuda.empty_cache()
    return score

# Reconstruction error
def recon_error(norm_data, G, Q, rank=None):
    G.eval()
    Q.eval()
    recon = []
    for dat in norm_data:
        with torch.no_grad():
            n = dat.shape[0]
            if rank is not None:
                recon_data = G(Q(dat, rank))
            else:
                recon_data = G(Q(dat))
            l2 = torch.linalg.norm((dat - recon_data).view(n, -1), dim=-1)
            recon += l2.cpu().numpy().tolist()
    mean, std = np.mean(recon), np.std(recon, ddof=1)
    print(f"Reconstruction error = {mean} ({std})")
    return mean, std

# MMD of latent code
def mmd_latent(norm_data, Q, rank=None):
    Q.eval()
    stats = []
    for dat in norm_data:
        with torch.no_grad():
            if rank is not None:
                z_hat = Q(dat, rank)
                z_hat = z_hat[:, :rank]
            else:
                z_hat = Q(dat)
            z = torch.randn_like(z_hat)
            mmd = mmd_penalty(z_hat, z, kernel="IMQ")
            stats.append(float(mmd))
    print(f"MMD = {np.mean(stats)}")
    return np.mean(stats)

# Real images
np.random.seed(123)
torch.manual_seed(123)
print("Real")
celeba = torch.load(path, map_location=dev, weights_only=True)
norm_data = []
real_data = []
real_data2 = []
for _ in range(nbatch):
    inds = torch.randperm(celeba.shape[0])[:bsize]
    norm_data.append(celeba[inds])
    img = 0.5 * (1.0 + celeba[inds])
    real_data.append(interpolate(img))
    inds = torch.randperm(celeba.shape[0])[:bsize]
    img = 0.5 * (1.0 + celeba[inds])
    real_data2.append(interpolate(img))
del celeba
torch.cuda.empty_cache()
# Inception score
inception_score(real_data)
# Fréchet inception distance
fid_score(real_data2, real_data)

# WGAN
np.random.seed(123)
torch.manual_seed(123)
print("\nWGAN")
config = "Z_1_128_SD_64_EP_100k_BS_128_LR_2_WD_0_SC_1_IT_1_10_LG_20-0_LM_0-0_LR_0-01"
path = f"./outputs/WGAN/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, weights_only=True))
netG.eval()
fake_data = gen_data(netG, z_dim, bsize, nbatch)
# Inception score
inception_score(fake_data)
# Fréchet inception distance
fid_score(fake_data, real_data)

# WAE
np.random.seed(123)
torch.manual_seed(123)
print("\nWAE")
config = "Z_1_128_SD_64_EP_100k_BS_128_LR_5_WD_0_SC_1_IT_1_10_LG_5-0_LM_100-0_LR_0-01"
path = f"./outputs/WAE/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, weights_only=True))
netG.eval()
path = f"./outputs/WAE/CelebA/{config}/netQ.pt"
netQ = Q_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netQ.load_state_dict(torch.load(path, weights_only=True))
netQ.eval()
# Generate images
fake_data = gen_data(netG, z_dim, bsize, nbatch)
# Inception score
inception_score(fake_data)
# Fréchet inception distance
fid_score(fake_data, real_data)
# Reconstruction error
recon_error(norm_data, netG, netQ)
# MMD of latent code
mmd_latent(norm_data, netQ)

# CycleGAN
np.random.seed(123)
torch.manual_seed(123)
print("\nCycleGAN")
config = "Z_1_128_SD_64_EP_10k_BS_128_LR_1_WD_0_SC_1_IT_5_15_LG_5-0_LM_0-0_LR_0-01"
path = f"./outputs/CycleGAN/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, weights_only=True))
netG.eval()
path = f"./outputs/CycleGAN/CelebA/{config}/netQ.pt"
netQ = Q_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netQ.load_state_dict(torch.load(path, weights_only=True))
netQ.eval()
# Generate images
fake_data = gen_data(netG, z_dim, bsize, nbatch)
# Inception score
inception_score(fake_data)
# Fréchet inception distance
fid_score(fake_data, real_data)
# Reconstruction error
recon_error(norm_data, netG, netQ)
# MMD of latent code
mmd_latent(norm_data, netQ)

# LWGAN
np.random.seed(123)
torch.manual_seed(123)
config = "Z_1_128_SD_64_EP_100k_BS_128_LR_1_WD_0_SC_1_IT_5_10_LG_5-0_LM_1-0_LR_0-002"
path = f"./outputs/CelebA/{config}/netG.pt"
netG = G_CelebA(z_dim=z_dim, struct_dim=64, act="relu").to(device=dev)
netG.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
netG.eval()
path = f"./outputs/CelebA/{config}/netQ.pt"
netQ = Qrank_CelebA(z_dim=z_dim, rank_min=1, struct_dim=64, act="relu").to(device=dev)
netQ.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
netQ.eval()

print("\nLWGAN-16")
fake_data = gen_data(netG, z_dim, bsize, nbatch, rank=16)
# Inception score
inception_score(fake_data)
# Fréchet inception distance
fid_score(fake_data, real_data)
# Reconstruction error
recon_error(norm_data, netG, netQ, rank=16)
# MMD of latent code
mmd_latent(norm_data, netQ, rank=16)

print("\nLWGAN-34")
fake_data = gen_data(netG, z_dim, bsize, nbatch, rank=34)
# Inception score
inception_score(fake_data)
# Fréchet inception distance
fid_score(fake_data, real_data)
# Reconstruction error
recon_error(norm_data, netG, netQ, rank=34)
# MMD of latent code
mmd_latent(norm_data, netQ, rank=34)

print("\nLWGAN-64")
fake_data = gen_data(netG, z_dim, bsize, nbatch, rank=64)
# Inception score
inception_score(fake_data)
# Fréchet inception distance
fid_score(fake_data, real_data)
# Reconstruction error
recon_error(norm_data, netG, netQ, rank=64)
# MMD of latent code
mmd_latent(norm_data, netQ, rank=64)

print("\nLWGAN-128")
fake_data = gen_data(netG, z_dim, bsize, nbatch, rank=128)
# Inception score
inception_score(fake_data)
# Fréchet inception distance
fid_score(fake_data, real_data)
# Reconstruction error
recon_error(norm_data, netG, netQ, rank=128)
# MMD of latent code
mmd_latent(norm_data, netQ, rank=128)
