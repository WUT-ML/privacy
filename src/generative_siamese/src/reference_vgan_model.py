"""Privacy-Preserving Representation-Learning Variational Generative Adversarial Network model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


class Generator(nn.Module):
    """Generator (forger) neural network."""

    def __init__(self, conv_size=32, kernel=5, stride=2):
        """Initialize generator."""
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, conv_size, kernel, stride)
        self.conv1_bn = nn.BatchNorm2d(conv_size)
        self.conv2 = nn.Conv2d(conv_size, conv_size * 2, kernel, stride)
        self.conv2_bn = nn.BatchNorm2d(conv_size * 2)
        self.conv3 = nn.Conv2d(conv_size * 2, conv_size * 4, kernel, stride)
        self.conv3_bn = nn.BatchNorm2d(conv_size * 4)
        self.conv4 = nn.Conv2d(conv_size * 4, conv_size * 8, kernel, stride)
        self.conv4_bn = nn.BatchNorm2d(conv_size * 8)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(conv_size * 4, conv_size * 8, kernel, stride)
        self.deconv1_bn = nn.BatchNorm2d(conv_size * 8)
        self.deconv2 = nn.ConvTranspose2d(conv_size * 8, conv_size * 4, kernel, stride)
        self.deconv2_bn = nn.BatchNorm2d(conv_size * 4)
        self.deconv3 = nn.ConvTranspose2d(conv_size * 4, conv_size * 2, kernel, stride)
        self.deconv3_bn = nn.BatchNorm2d(conv_size * 2)
        self.deconv4 = nn.ConvTranspose2d(conv_size * 2, 1, kernel, stride)

        self.weight_init()

    def weight_init(self, mean=0.0, std=0.01):
        """Initilize weights."""
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def encode(self, input):
        """Encode image into latent vector."""
        c1 = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
        c2 = F.leaky_relu(self.conv2_bn(self.conv2(c1)), 0.2)
        c3 = F.leaky_relu(self.conv3_bn(self.conv3(c2)), 0.2)
        c4 = F.leaky_relu(self.conv4_bn(self.conv4(c3)), 0.2)
        fc11 = nn.Linear(c4.shape[1] * c4.shape[2] * c4.shape[3], 128).cuda()  # mu layer
        fc12 = nn.Linear(c4.shape[1] * c4.shape[2] * c4.shape[3], 128).cuda()  # logvar layer

        mu = fc11(c4.view(c4.size(0), -1))
        logvar = fc12(c4.view(c4.size(0), -1))
        return mu, logvar

    def decode(self, z, label):
        """Decode image from latent vector."""
        input = torch.cat([z, label], 1)
        fc2 = nn.Linear(input.shape[1], 2048).cuda()
        decoder_in = F.leaky_relu(fc2(input).view(-1, 128, 4, 4), 0.2)
        d1 = F.leaky_relu(self.deconv1_bn(self.deconv1(decoder_in)), 0.2)
        d2 = F.leaky_relu(self.deconv2_bn(self.deconv2(d1)), 0.2)
        d3 = F.leaky_relu(self.deconv3_bn(self.deconv3(d2)), 0.2)
        decoder_out = F.tanh(self.deconv4(d3))
        return decoder_out

    def reparameterize(self, mu, logvar):
        """Z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = Variable(torch.randn(mu.size(0), mu.size(1)))
        eps = eps.cuda()
        z = mu + eps * torch.exp(logvar/2)    # 2 for convert var to std
        return z

    def forward(self, input, label):
        """Define the computation performed at every call."""
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, label), mu, logvar


class Discriminator(nn.Module):
    """Discriminator (detective) neural network."""

    def __init__(self, n_classes, n_attributes, conv_size=32, kernel=5, stride=2):
        """Initialize discriminator."""
        super(Discriminator, self).__init__()

        self.n_classes = n_classes
        self.n_attributes = n_attributes

        self.conv1 = nn.Conv2d(1, conv_size, kernel, stride)
        self.conv1_bn = nn.BatchNorm2d(conv_size)
        self.conv2 = nn.Conv2d(conv_size, conv_size * 2, kernel, stride)
        self.conv2_bn = nn.BatchNorm2d(conv_size * 2)
        self.conv3 = nn.Conv2d(conv_size * 2, conv_size * 4, kernel, stride)
        self.conv3_bn = nn.BatchNorm2d(conv_size * 4)
        self.conv4 = nn.Conv2d(conv_size * 4, conv_size * 8, kernel, stride)
        self.conv4_bn = nn.BatchNorm2d(conv_size * 8)
        self.softmax_identity = nn.Softmax()
        self.softmax_attribute = nn.Softmax()

        self.weight_init()

    def weight_init(self, mean=0.0, std=0.01):
        """Initilize weights."""
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        """Define the computation performed at every call."""
        c1 = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
        c2 = F.leaky_relu(self.conv2_bn(self.conv2(c1)), 0.2)
        c3 = F.leaky_relu(self.conv3_bn(self.conv3(c2)), 0.2)
        c4 = F.leaky_relu(self.conv4_bn(self.conv4(c3)), 0.2)
        fc1 = nn.Linear(c4.shape[1] * c4.shape[2] * c4.shape[3], 256).cuda()
        fc_fake = nn.Linear(256, 1).cuda()
        fc_identity = nn.Linear(256, self.n_classes).cuda()
        fc_attribute = nn.Linear(256, self.n_attributes).cuda()
        discriminator_out_common = F.leaky_relu(fc1(c4.view(c4.size(0), -1)), 0.2)
        discriminator_out_fake = fc_fake(discriminator_out_common)
        discriminator_out_identity = fc_identity(discriminator_out_common)
        discriminator_out_attribute = fc_attribute(discriminator_out_common)

        return (discriminator_out_fake,
                self.softmax_identity(discriminator_out_identity),
                self.softmax_attribute(discriminator_out_attribute))


def normal_init(m, mean, std):
    """Initialize conv layer weights using normal distribution."""
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
