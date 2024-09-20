## LWGAN

This folder contains the code to reproduce the experiments in the article
*Adaptive Learning of the Latent Space of Wasserstein Generative Adversarial Networks*.

### Structure

The structure of this repository is as follows:
```
<working directory>/
├─ data/
├─ models/
├─ utils/
├─ fig_celeba.py
├─ fig_dim_mismatch.py
├─ fig_mnist.py
├─ fig_toy.py
├─ run_celeba.sh
├─ run_mnist.sh
├─ run_toy.sh
├─ run_toy_bootstrap.sh
├─ tab_celeba_score.py
├─ train_celeba.py
├─ train_celeba.cyclegan.py
├─ train_celeba_wae.py
├─ train_celeba_wgan.py
├─ train_mnist.py
├─ train_toy.py
├─ train_toy_bootstrap.py
├─ README.md
```

- `data`: the directory containing the preprocessed CelebA dataset. See the **Preparations** section for details.
- `models`: core PyTorch code to implement various generative models.
- `utils`: utility functions for running experiments.
- `train_.*py`: code for training generative models on various datasets.
- `run_*.sh`: scripts to train models and save output results.
- `fig_*.py`: code to generate figures in the main article.
- `tab_celeba_score.py`: code to generate table outputs.
- `README.md`: this document.

### Workflow

#### Preparations

1. Install the PyTorch framework, following the installation guides at https://pytorch.org/get-started/locally/.
2. Install necessary Python packages as in the **Software Environment** section.
3. Download the `data` folder from https://drive.google.com/drive/folders/1vi639Nbkg_AAbr0i3-hKeWCOulQg-ItU to the local working directory, assuming the following directory structure:
```
<working directory>/
├─ data/
├─ models/
├─ utils/
├─ ...
├─ README.md
```

#### Toy Examples

1. The toy example data are simulated by the `inf_train_gen()` function in the `utils/tools.py` file.
2. Run the `run_toy.sh` script in the working directory to train LWGAN models on the simulated data:
```
./run_toy.sh
```
3. Run the `run_toy_bootstrap.sh` script to conduct the bootstrap experiments:
```
./run_toy_bootstrap.sh
```
4. After the two scripts finish, a directory named `outputs` will be created in the working directory, with the following structure:
```
outputs/
├─ hyperplane/
│  ├─ Z_7_QGD_64_64_64_EP_10k_BS_512_LR_2_SC_1_IT_1_20_LG_5-0_LM_1-0_LR_0-01/
│  │  ├─ bootstrap/
├─ scurve/
│  ├─ Z_5_QGD_64_64_64_EP_10k_BS_512_LR_2_SC_1_IT_1_20_LG_5-0_LM_1-0_LR_0-01/
│  │  ├─ bootstrap/
├─ swillroll/
│  ├─ Z_5_QGD_64_64_64_EP_5k_BS_512_LR_2_SC_1_IT_5_20_LG_5-0_LM_1-0_LR_0-01/
│  │  ├─ bootstrap/
```
5. Run the `fig_dim_mismatch.py` and `fig_toy.py` files to generate plots in the main article related to the toy examples:
```
python fig_dim_mismatch.py
python fig_toy.py
```
6. After the two scripts above finish, a directory named `plots` will be created in the working directory, containing the following files:
```
plots/
├─ fig1_real.pdf               (Figure 1)
├─ fig1_wae_gen.pdf            (Figure 1)
├─ fig1_wae_latent.pdf         (Figure 1)
├─ fig1_wgan_gen.pdf           (Figure 1)
├─ hyperplane.png              (Figure 2)
├─ hyperplane_gen.png          (Figure 2)
├─ hyperplane_rank.pdf         (Figure 2)
├─ hyperplane_recon.png        (Figure 2)
├─ scurve.pdf                  (Figure 2)
├─ scurve_gen.pdf              (Figure 2)
├─ scurve_rank.pdf             (Figure 2)
├─ scurve_recon.pdf            (Figure 2)
├─ swissroll.pdf               (Figure 2)
├─ swissroll_gen.pdf           (Figure 2)
├─ swissroll_rank.pdf          (Figure 2)
├─ swissroll_recon.pdf         (Figure 2)
```
In the parentheses we show the figure in the main article that each file is used for.

#### MNIST Data

1. The MNIST data can be loaded by the `torchvision.datasets.MNIST()` function. It will automatically download the data files in the first run.
2. Run the `run_mnist.sh` script in the working directory to train LWGAN models on the MNIST data:
```
./run_mnist.sh
```
3. After the script finishes, the following subdirectories will be created under the `outputs` directory:
```
outputs/
├─ MNIST/
├─ MNIST_digit_1/
├─ MNIST_digit_2/
```
4. Run the `fig_mnist.py` file to generate plots in the main article related to the MNIST data:
```
python fig_mnist.py
```
5. After the script finishes, the following files will be generated under the `plots` directory:
```
plots/
├─ mnist_digit_1.pdf              (Figure 3)
├─ mnist_digit_1_gen.pdf          (Figure 3)
├─ mnist_digit_1_rank.pdf         (Figure 3)
├─ mnist_digit_1_recon.pdf        (Figure 3)
├─ mnist_digit_2.pdf              (Figure 3)
├─ mnist_digit_2_gen.pdf          (Figure 3)
├─ mnist_digit_2_rank.pdf         (Figure 3)
├─ mnist_digit_2_recon.pdf        (Figure 3)
├─ mnist.pdf                      (Figure 4)
├─ mnist_gen.pdf                  (Figure 4)
├─ mnist_interp.pdf               (Figure 4)
├─ mnist_rank.pdf                 (Figure 4)
├─ mnist_recon.pdf                (Figure 4)
```
In the parentheses we show the figure in the main article that each file is used for.

#### CelebA Data

1. The preprocessed CelebA data are contained in the `data/face64.pt` PyTorch tensor file. See the document `data/README.md` for how this file is generated.
2. Run the `run_celeba.sh` script in the working directory to train LWGAN models on the CelebA data:
```
./run_celeba.sh
```
3. After the script finishes, the following subdirectories will be created under the `outputs` directory:
```
outputs/
├─ CelebA/
├─ CycleGAN/
│  ├─ CelebA/
├─ WAE/
│  ├─ CelebA/
├─ WGAN/
│  ├─ CelebA/
```
4. Run the `fig_celeba.py` file to generate plots in the main article related to the CelebA data:
```
python fig_celeba.py
```
5. After the script finishes, the following files will be generated under the `plots` directory:
```
plots/
├─ celeba_wide.jpg                   (Figure 5)
├─ celeba_rank.pdf                   (Figure 5)
├─ celeba_gen_cyclegan.jpg           (Figure 6)
├─ celeba_gen_lwgan16.jpg            (Figure 6)
├─ celeba_gen_lwgan32.jpg
├─ celeba_gen_lwgan34.jpg            (Figure 6)
├─ celeba_gen_lwgan64.jpg
├─ celeba_gen_lwgan128.jpg           (Figure 6)
├─ celeba_gen_wae.jpg                (Figure 6)
├─ celeba_gen_wgan.jpg               (Figure 6)
├─ celeba.jpg                        (Figure 7)
├─ celeba_recon_cyclegan.jpg         (Figure 7)
├─ celeba_recon_lwgan16.jpg
├─ celeba_recon_lwgan32.jpg
├─ celeba_recon_lwgan34.jpg          (Figure 7)
├─ celeba_recon_lwgan64.jpg
├─ celeba_recon_lwgan128.jpg
├─ celeba_recon_wae.jpg              (Figure 7)
├─ celeba_interp_cyclegan.jpg        (Figure 8)
├─ celeba_interp_lwgan34.jpg         (Figure 8)
├─ celeba_interp_wae.jpg             (Figure 8)
├─ celeba_losses.pdf                 (Figure S1)
```
6. Finally, run the `tab_celeba_score.py` file to output the values in Table 1 of the main article:
```
python tab_celeba_score.py
```

### Software Environment

GPU computing:

- CUDA 12.1

Python:

- Python 3.12.5
- Numpy 1.26.4
- SciPy 1.13.1
- Pandas 2.2.2
- Tqdm 4.66.5
- Pillow 10.4.0
- OpenCV 4.10.0.84
- Matplotlib 3.9.2
- Seaborn 0.13.2
- PyTorch 2.3.1
- TorchMetrics 1.4.2
- TorchVision 0.18.1

The commands below create a virtual environment named `pytorch` in the Conda environment,
and then install the packages above.

```bash
conda create -n pytorch
conda activate pytorch
conda install python=3.12 numpy scipy pandas tqdm pillow matplotlib seaborn
conda install pytorch=2.3 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python torchmetrics[image]
```
