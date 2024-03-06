# Library imports
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
from PIL import Image

from utils import ImageDataset

# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
random.seed(seed)
#
# Automatically get the current user's home directory
base_dir = os.getcwd()


train_path = os.path.join(base_dir, "data/trainA")
test_path = os.path.join(base_dir, "data/testA")

# Create a directory to save generated images
images_dir = os.path.join(base_dir, "generated_images")
os.makedirs(images_dir, exist_ok=True)


image_size = 128  # Height / width of the square images
batch_size = 1

transform = transforms.Compose([transforms.ToTensor()])

transforms_ = [
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    transforms.Resize((image_size, image_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust normalization for a single channel
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

# Create dataset instances
train_dataset = ImageDataset(train_path, transforms_=transforms_)
test_dataset = ImageDataset(test_path, transforms_=transforms_)

# Create DataLoader for train and test sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        input_features = image_size[0] * image_size[1] * 1 # Assuming grayscale for simplicity; adjust if RGB

        self.model = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128 * 4),  # Upscale noise input
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128 * 4, 64 * 32 * 32),  # Further upscale to prepare for image size
            nn.BatchNorm1d(64 * 32 * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64 * 32 * 32, 1 * 128 * 128),  # Target image size
            nn.Tanh()  # Normalize the output to [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 128, 128)  # Reshape to image dimensions
        return img

class PatchClassicalGenerator(nn.Module):
    """Classical generator class for the patch method"""

    def __init__(self, n_generators, patch_size=8, noise_dim=100):
        """
        Args:
            n_generators (int): Number of sub-generators (patches) to be used in the patch method.
            patch_size (int): Size of the output from each sub-generator, assuming square patches.
            noise_dim (int): The dimensionality of the input noise vector.
        """

        super(PatchClassicalGenerator, self).__init__()

        self.n_generators = n_generators
        self.patch_size = patch_size
        self.noise_dim = noise_dim

        # Define a list of sub-generators, each is a simple neural network
        self.sub_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(noise_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, patch_size * patch_size),
                    nn.Tanh()  # Use Tanh to keep output values in a reasonable range
                )
                for _ in range(n_generators)
            ]
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Create an empty tensor to accumulate the generated image patches
        #images = torch.Tensor(batch_size, 0, self.patch_size, self.patch_size).to(x.device)
        images = torch.Tensor(x.size(0), 0).to(device)
        # Iterate over all sub-generators
        for gen in self.sub_generators:
            # Generate a patch for each element in the batch
            gen_noise = torch.randn(batch_size, self.noise_dim).to(x.device)
            patch = gen(gen_noise)
            patch = patch.view(batch_size, 1, self.patch_size, self.patch_size)  # Reshape to the correct patch size
            #patch = torch.cat((patches, patch))
            # Concatenate the generated patches to form the full image
            #images = torch.cat((images, patch), 2)  # Adjust the dimension as necessary
            images = torch.cat((images, patch), 1)

        # Assuming square images for simplicity; adjust if necessary for other shapes
        n_patches_side = int(math.sqrt(self.n_generators))
        images = images.view(batch_size, 1, n_patches_side * self.patch_size, n_patches_side * self.patch_size)

        return images



lrG = 0.3  # Learning rate for the generator
lrD = 0.01  # Learning rate for discriminator
num_iter = 2000  # Number of training iterations

n_qubits = 8  # 5  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 128 #4  # Number of subgenerators for the patch method / N_G
patch_size = 32  # Size of each patch assuming square patches

# Enable CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming `train_dataset` is an instance of `ImageDataset` and has an attribute `image_size`
#discriminator = Discriminator(image_size=train_dataset.image_size)
discriminator = Discriminator(image_size=[128, 128]).to(device)
# Instantiate the classical generator.
"""generator = PatchClassicalGenerator(n_generators=n_generators,
                                    patch_size= 2 ** (n_qubits - n_a_qubits)
                                ).to(device)"""
generator = Generator().to(device)


# Binary cross entropy
criterion = nn.BCELoss()

# Optimisers
optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Fixed noise allows us to visually track the generated images throughout training
#fixed_noise = torch.rand(8, n_qubits , device=device) * math.pi / 2
fixed_noise = torch.randn(8, 100, device=device)  # Adjust to the correct size for visualization


# Iteration counter
counter = 0

# Collect images for plotting later
results = []

while True:
    for i, data in enumerate(train_dataloader):
        print(" Running ..")

        # Data for training the discriminator
        # data = data.reshape(-1, image_size * image_size)
        data = data.view(data.size(0), -1)
        real_data = data.to(device)

        # Noise follwing a uniform distribution in range [0,pi/2)
        #noise = torch.rand(batch_size, n_qubits , device=device) * math.pi / 2
        noise_dim = 100  # This should match the generator's expected noise dimension
        noise = torch.randn(batch_size, noise_dim, device=device)
        fake_data = generator(noise)

        # Training the discriminator
        discriminator.zero_grad()
        #outD_real = discriminator(real_data)
        #outD_fake = discriminator(fake_data.detach())
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)
        # Propagate gradients
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        optD.step()

        # Training the generator
        generator.zero_grad()
        outD_fake = discriminator(fake_data).view(-1)
        errG = criterion(outD_fake, real_labels)
        errG.backward()
        optG.step()

        counter += 1

        # Show loss values
        if counter % 10 == 0:
            print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
            test_images = generator(fixed_noise).view(8, 1, image_size, image_size).cpu().detach()

        # Save images every 50 iterations
        if counter % 50 == 0:
            #test_images = generator(fixed_noise).view(8, 1, image_size, image_size).cpu().detach()
            test_images = generator(fixed_noise).view(8, 1, image_size, image_size).cpu().detach()
            results.append(test_images)

            # Save the generated images
            for idx, image in enumerate(test_images):
                save_path = os.path.join(images_dir, f'image_{counter}_{idx}.png')
                torchvision.utils.save_image(image, save_path)

        if counter == num_iter:
            break
    if counter == num_iter:
        break

fig = plt.figure(figsize=(10, 5))
outer = gridspec.GridSpec(5, 2, wspace=0.1)

"""for i, images in enumerate(results):
    inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
                                             subplot_spec=outer[i])
    images = torch.squeeze(images, dim=1)
    for j, im in enumerate(images):
        ax = plt.Subplot(fig, inner[j])
        ax.imshow(im.numpy(), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_title(f'Iteration{50 + i * 50}', loc='left')
        fig.add_subplot(ax)

plt.show()"""


