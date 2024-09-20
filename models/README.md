This folder contains the code to define the encoders, generators, discriminators,
and loss functions of various deep generative models.

### Structure

- `blocks.py`: Building blocks of neural network structures.
- `losses.py`: The code used to define various loss functions.
- `wae.py`: Implementation of the WAE model based on the article [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558).
- `wgan.py`: Implementation of the WGAN model based on the article [Wasserstein Generative Adversarial Networks](https://proceedings.mlr.press/v70/arjovsky17a.html).
- `cyclegan.py`: Implementation of the CycleGAN model based on the article [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).
- `lwgan.py`: Implementation of the LWGAN model proposed in this article.
- `toy.py`: The encoder, generator, and discriminator for the toy examples.
- `mnist.py`: The encoder, generator, and discriminator for the MNIST examples.
- `celeba.py`: The encoder, generator, and discriminator for the CelebA examples.
- `README.md`: This document.
