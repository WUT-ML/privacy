"""Deep convolutional GAN model."""
import torch.nn as nn
import torch.nn.functional as F


def deconv(ch_in, ch_out, kernel_size, stride=2, padding=1, batch_norm=True):
    """Deconvolutional layer with optional batch normalization."""
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding))
    if batch_norm:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)


def conv(ch_in, ch_out, kernel_size, stride=2, padding=1, batch_norm=True):
    """Convolutional layer with optional batch normalization."""
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding))
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
        self.deconv4 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, z):
        """Define the computation performed at every call."""
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.tanh(self.deconv4(out))
        return out


class Discriminator(nn.Module):
    """Discriminator (detective) neural network."""

    def __init__(self, image_size, conv_dim):
        """Set parameters of discriminator neural network."""
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)

    def forward(self, x):
        """Define the computation performed at every call."""
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = self.fc(out).squeeze()
        return out
