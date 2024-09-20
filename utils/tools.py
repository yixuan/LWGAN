import random
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from models.blocks import gen_noise_with_rank

# Various random seeds
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)

# Generate directory name for output files -- toy examples
def filename_toy(args):
    return "Z_" + str(args.z_dim) + \
           "_QGD_" + str(args.q_dim) + "_" + str(args.g_dim) + "_" + str(args.d_dim) +\
           "_EP_" + str(args.epochs // 1000) + "k" + \
           "_BS_" + str(args.batch_size) + \
           "_LR_" + str(math.ceil(args.learning_rate * 1e4)) + \
           "_SC_" + str(int(args.scheduler)) + \
           "_IT_" + str(args.iter_gq) + "_" + str(args.iter_d) + \
           "_LG_" + str(args.lambda_gp).replace(".", "-") + \
           "_LM_" + str(args.lambda_mmd).replace(".", "-") + \
           "_LR_" + str(args.lambda_rank).replace(".", "-")

# Generate directory name for output files -- MNIST data
def filename_mnist(args):
    return "Z_" + str(args.z_dim_min) + "_" + str(args.z_dim_max) + \
           "_SD_" + str(args.structure_dim) + \
           "_EP_" + str(args.epochs // 1000) + "k" + \
           "_BS_" + str(args.batch_size) + \
           "_LR_" + str(math.ceil(args.learning_rate * 1e4)) + \
           "_WD_" + str(math.ceil(args.weight_decay * 1e4)) + \
           "_SC_" + str(int(args.scheduler)) + \
           "_IT_" + str(args.iter_gq) + "_" + str(args.iter_d) + \
           "_LG_" + str(args.lambda_gp).replace(".", "-") + \
           "_LM_" + str(args.lambda_mmd).replace(".", "-") + \
           "_LR_" + str(args.lambda_rank).replace(".", "-")

# Generate directory name for output files -- CelebA data
def filename_celeba(args):
    return "Z_" + str(args.z_dim_min) + "_" + str(args.z_dim_max) + \
           "_SD_" + str(args.structure_dim) + \
           "_EP_" + str(args.epochs // 1000) + "k" + \
           "_BS_" + str(args.batch_size) + \
           "_LR_" + str(math.ceil(args.learning_rate * 1e4)) + \
           "_WD_" + str(math.ceil(args.weight_decay * 1e4)) + \
           "_SC_" + str(int(args.scheduler)) + \
           "_IT_" + str(args.iter_gq) + "_" + str(args.iter_d) + \
           "_LG_" + str(args.lambda_gp).replace(".", "-") + \
           "_LM_" + str(args.lambda_mmd).replace(".", "-") + \
           "_LR_" + str(args.lambda_rank).replace(".", "-")

# Toy example data simulator
def inf_train_gen(dataset, batch_size=1024, std=0.02):
    if dataset == "swissroll":
        while True:
            delta_theta = 1.5 * np.pi * np.random.uniform(1.0, 3.0, size=batch_size)
            p1 = delta_theta * np.cos(delta_theta)
            p2 = delta_theta * np.sin(delta_theta)
            dataset = 0.2 * np.stack((p1, p2), axis=1).astype(dtype="float32")
            dataset += np.random.normal(0.0, std, size=(batch_size, 2))
            yield dataset
    if dataset == "scurve":
        while True:
            delta_theta = 3 * np.pi * (np.random.uniform(-0.5, 0.5, size=batch_size))
            p1 = np.sin(delta_theta)
            p2 = np.random.uniform(0.0, 2.0, size=batch_size)
            p3 = np.sign(delta_theta) * (np.cos(delta_theta) - 1.0)
            dataset = np.stack((p1, p2, p3), axis=1).astype(dtype="float32")
            yield dataset
    if dataset == "hyperplane":
        while True:
            point = np.random.normal(size=(batch_size, 5))
            point[:, 4] = point[:, 0] + point[:, 1] + point[:, 2] + point[:, 3]**2
            dataset = point.astype(dtype="float32")
            yield dataset

# Load MNIST data
def mnist_loader(batch_size, shuffle_train=True, shuffle_test=False, digit=None, test_batch_size=None, device=torch.device("cpu")):
    mnist_train = datasets.MNIST("./outputs/data/MNIST", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./outputs/data/MNIST", train=False, download=True, transform=transforms.ToTensor())
    if digit is not None:
        idx_train = (mnist_train.targets == digit)
        mnist_train.data = mnist_train.data[idx_train]
        idx_test = (mnist_test.targets == digit)
        mnist_test.data = mnist_test.data[idx_test]
    data_train = 0.001 + 0.998 * mnist_train.data / 255.0
    data_train = data_train.to(device=device).view(-1, 1, 28, 28)
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    if test_batch_size is not None:
        batch_size = test_batch_size
    data_test = 0.001 + 0.998 * mnist_test.data / 255.0
    data_test = data_test.to(device=device).view(-1, 1, 28, 28)
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=shuffle_test, drop_last=True)
    return train_loader, test_loader

# Various visualization functions
def plot2d_gen(netG, nsamp, zdim, rank, device, xlim=None, ylim=None, filename=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    fig = plt.figure(figsize=(7, 7))
    with torch.no_grad():
        noise = gen_noise_with_rank(nsamp, zdim, rank, device)
        fd = netG(noise).detach().cpu().numpy()
    plt.scatter(fd[:, 0], fd[:, 1], label="Generated Samples by Z")
    plt.title("Generated samples: G(Z)")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close()

def plot3d_gen(netG, nsamp, zdim, rank, device, filename=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection="3d")
    with torch.no_grad():
        noise = gen_noise_with_rank(nsamp, zdim, rank, device)
        fd = netG(noise).detach().cpu().numpy()
    ax.scatter(fd[:, 0], fd[:, 1], fd[:, 2], c=fd[:, 2], cmap=plt.cm.Spectral, label="Generated Samples by z")
    ax.view_init(15, -110)
    ax.set_title("Generated samples: G(Z)")
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close()

def plotrd_gen(netG, nsamp, zdim, rank, device, filename=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    fig = plt.figure(figsize=(15, 15))
    with torch.no_grad():
        noise = gen_noise_with_rank(nsamp, zdim, rank, device)
        fd = netG(noise).detach().cpu().numpy()
    xdim = fd.shape[1]
    for i in range(xdim):
        for j in range(xdim):
            if i != j:
                plt.subplot(xdim, xdim, i * xdim + j + 1)
                plt.scatter(fd[:, i], fd[:, j], marker="*", alpha=0.5)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close()

def plot_latent(netQ, rank, x, filename=None):
    with torch.no_grad():
        z = netQ(x, rank)
    zdim = z.size(1)
    zdim = min(zdim, 10)
    fig = plt.figure(figsize=(15, 15))
    latent_Q = z.cpu().numpy()
    for i in range(zdim):
        for j in range(zdim):
            if i != j:
                plt.subplot(zdim, zdim, i * zdim + j + 1)
                plt.scatter(latent_Q[:, i], latent_Q[:, j], marker="*", alpha=0.5)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        plt.close()
