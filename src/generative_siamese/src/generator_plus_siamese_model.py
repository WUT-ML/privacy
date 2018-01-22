"""GAN-like model with siamese discriminator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def deconv(ch_in, ch_out, kernel_size, stride=2, padding=1, batch_norm=True):
    """Deconvolutional layer with optional batch normalization."""
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding))
    if batch_norm:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """Generator (forger) neural network."""

    def __init__(self, noise_dim, image_size, conv_dim):
        """Set parameters of generator neural network."""
        super(Generator, self).__init__()
        self.fc = deconv(noise_dim, conv_dim*8, int(image_size/16), 1, 0, batch_norm=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 1, 4, batch_norm=False)

    def forward(self, z, image):
        """Define the computation performed at every call."""
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.tanh(self.deconv4(out))
        out = out + image
        return out


class SiameseDiscriminator(nn.Module):
    """Discriminator neural network, using siamese networks."""

    def __init__(self, image_size):
        """Set parameters of discriminator neural network."""
        super(SiameseDiscriminator, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2))

        self.fc1 = nn.Sequential(
            nn.Linear(8 * image_size * image_size, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 15))

    def forward_once(self, x):
        """Define the computation performed at every call by one side of siamese network."""
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        """Define the computation performed at every call."""
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        """Set parameters of contrastive loss function."""
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, max_L2=None, original_images=None, fake_images=None):
        """Define the computation performed at every call."""
        # Check constraint, if fake image does not differ too much from original image
        if(original_images is not None and fake_images is not None and max_L2 is not None):
            # Use L2 metric
            distance = distanceL2(original_images, fake_images)
            if ((distance.data[0]) > max_L2):
                return distance

        # Otherwise use contrastive loss function
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) *
                                      torch.pow(euclidean_distance, 2) +
                                      label *
                                      torch.pow(torch.clamp(self.margin -
                                                            euclidean_distance, min=0.0), 2))
        return loss_contrastive


def distanceL2(image1, image2):
    """L2 distance metric."""
    return torch.sum((image1-image2)**2)
