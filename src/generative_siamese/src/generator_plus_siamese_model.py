"""GAN-like model with siamese discriminator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


class Generator(nn.Module):
    """Generator (forger) neural network."""

    def __init__(self, conv_size, kernel=4, stride=2, pad=1):
        """Initialize generator."""
        super(Generator, self).__init__()
        self.NOISE_FACTOR = 0.01

        # Unet encoder
        self.conv1 = nn.Conv2d(1, conv_size, kernel, stride, pad)
        self.conv2 = nn.Conv2d(conv_size, conv_size * 2, kernel, stride, pad)
        self.conv2_bn = nn.BatchNorm2d(conv_size * 2)
        self.conv3 = nn.Conv2d(conv_size * 2, conv_size * 4, kernel, stride, pad)
        self.conv3_bn = nn.BatchNorm2d(conv_size * 4)
        self.conv4 = nn.Conv2d(conv_size * 4, conv_size * 8, kernel, stride, pad)
        self.conv4_bn = nn.BatchNorm2d(conv_size * 8)
        self.conv5 = nn.Conv2d(conv_size * 8, conv_size * 8, kernel, stride, pad)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(conv_size * 8, conv_size * 8, kernel, stride, pad)
        self.deconv1_bn = nn.BatchNorm2d(conv_size * 8)
        self.deconv2 = nn.ConvTranspose2d(conv_size * 8 * 2, conv_size * 4, kernel, stride, pad)
        self.deconv2_bn = nn.BatchNorm2d(conv_size * 4)
        self.deconv3 = nn.ConvTranspose2d(conv_size * 4 * 2, conv_size * 2, kernel, stride, pad)
        self.deconv3_bn = nn.BatchNorm2d(conv_size * 2)
        self.deconv4 = nn.ConvTranspose2d(conv_size * 2 * 2, conv_size, kernel, stride, pad)
        self.deconv4_bn = nn.BatchNorm2d(conv_size)
        self.deconv5 = nn.ConvTranspose2d(conv_size * 2, 1, kernel, stride, pad)

    def weight_init(self, mean, std):
        """Initilize weights."""
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        """Define the computation performed at every call."""
        c1 = self.conv1(input)
        c2 = self.conv2_bn(self.conv2(F.leaky_relu(c1, 0.1)))
        c3 = self.conv3_bn(self.conv3(F.leaky_relu(c2, 0.1)))
        c4 = self.conv4_bn(self.conv4(F.leaky_relu(c3, 0.1)))
        c5 = self.conv5(F.leaky_relu(c4, 0.1))

        # Add noise
        c5 += Variable((self.NOISE_FACTOR * (-0.5 + torch.rand(c5.size()))).cuda())

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.leaky_relu(c5, 0.1))), 0.5, training=True)
        d1 = torch.cat([d1, c4], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.leaky_relu(d1, 0.1))), 0.5, training=True)
        d2 = torch.cat([d2, c3], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.leaky_relu(d2, 0.1))), 0.5, training=True)
        d3 = torch.cat([d3, c2], 1)
        d4 = self.deconv4_bn(self.deconv4(F.leaky_relu(d3, 0.1)))
        d4 = torch.cat([d4, c1], 1)
        d5 = self.deconv5(F.leaky_relu(d4, 0.1))
        out = F.tanh(d5)

        return out


def normal_init(m, mean, std):
    """Initialize conv layer weights using normal distribution."""
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class SiameseDiscriminator(nn.Module):
    """Discriminator neural network, using siamese networks."""

    def __init__(self, image_size):
        """Set parameters of discriminator neural network."""
        super(SiameseDiscriminator, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2))

        self.fc1 = nn.Sequential(
            nn.Linear(8 * image_size * image_size, 500),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Linear(500, 500),
            nn.LeakyReLU(0.1, inplace=True),

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


class DistanceBasedLoss(torch.nn.Module):
    """
    Distance based loss function.

    For reference see:
    Hadsell et al., CVPR'06
    Chopra et al., CVPR'05
    Vo and Hays, ECCV'16
    """

    def __init__(self, margin):
        """Set parameters of distance-based loss function."""
        super(DistanceBasedLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """Define the computation performed at every call."""
        euclidean_distance = F.pairwise_distance(output1, output2)
        distance_from_margin = torch.clamp(torch.pow(euclidean_distance, 2) - self.margin, max=50.0)
        exp_distance_from_margin = torch.exp(distance_from_margin)
        distance_based_loss = (1.0 + math.exp(-self.margin)) / (1.0 + exp_distance_from_margin)
        similar_loss = -0.5 * (1 - label) * torch.log(distance_based_loss)
        dissimilar_loss = -0.5 * label * torch.log(1.0 - distance_based_loss)
        return torch.mean(similar_loss + dissimilar_loss)

    def predict(self, output1, output2, threshold_factor=0.5):
        """Predict a dissimilarity label given two embeddings.

        Return `1` if dissimilar.
        """
        return F.pairwise_distance(output1, output2) > self.margin * threshold_factor


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        """Set parameters of contrastive loss function."""
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """Define the computation performed at every call."""
        # Check constraint, if fake image does not differ too much from original image
        # if(original_images is not None and fake_images is not None and max_L2 is not None):
        #     # Use L2 metric
        #     distance = distanceL2(original_images, fake_images)
        #     if ((distance.data[0]) > max_L2):
        #         return distance

        euclidean_distance = F.pairwise_distance(output1, output2)
        clamped = torch.clamp(self.margin - euclidean_distance, min=0.0)
        similar_loss = (1 - label) * 0.5 * torch.pow(euclidean_distance, 2)
        dissimilar_loss = label * 0.5 * torch.pow(clamped, 2)
        contrastive_loss = similar_loss + dissimilar_loss

        return torch.mean(contrastive_loss)

    def predict(self, output1, output2, threshold_factor=0.5):
        """Predict a dissimilarity label given two embeddings.

        Return `1` if dissimilar.
        """
        return F.pairwise_distance(output1, output2) > self.margin * threshold_factor


def distanceL2(image1, image2):
    """L2 distance metric."""
    return torch.sum((image1-image2)**2)
