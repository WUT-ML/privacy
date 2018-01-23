"""Deep convolutional GAN with siamese discriminator."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from tqdm import tqdm

from generator_plus_siamese_model import Generator, SiameseDiscriminator, ContrastiveLoss


class SiameseGanSolver(object):
    """Solving GAN-like neural network with siamese discriminator."""

    def __init__(self, config, data_loader):
        """Set parameters of neural network and its training."""
        self.generator = None
        self.discriminator = None
        self.contrastive_loss = None

        self.g_optimizer = None
        self.d_optimizer = None

        self.g_conv_dim = 128
        self.noise_dim = 100
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.learning_rate = 0.0001
        self.image_size = config.image_size
        self.num_epochs = config.num_epochs
        self.max_L2 = config.max_L2

        self.data_loader = data_loader
        self.sample_path = config.sample_path
        self.model_path = config.model_path

        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        self.generator = Generator(self.noise_dim, self.image_size, self.g_conv_dim)
        self.discriminator = SiameseDiscriminator(self.image_size)
        self.contrastive_loss = ContrastiveLoss(2.0)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), self.learning_rate, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), self.learning_rate, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.contrastive_loss.cuda()

    def train(self):
        """Train generator and discriminator."""

        # First train discriminator on real images
        tqdm.write('Phase 1 (train discriminator to classify image pairs as same/different ID)')
        epoch_monitor = tqdm(total=self.num_epochs)
        epoch_monitor.set_description('Epoch')

        for epoch in range(self.num_epochs):
            batch_monitor = tqdm(total=len(self.data_loader))
            batch_monitor.set_description('Batch')

            for label, images0, images1 in self.data_loader:
                images0 = to_variable(images0)
                images1 = to_variable(images1)
                label = to_variable(label)

                # Train discriminator to recognize identity of real images
                output1, output2 = self.discriminator(images0, images1)
                d_real_loss = self.contrastive_loss(output1, output2, label)
                batch_monitor.set_postfix(d_real_loss=d_real_loss.data[0])

                # Backpropagation
                self.contrastive_loss.zero_grad()
                self.discriminator.zero_grad()
                d_real_loss.backward()
                self.d_optimizer.step()

                batch_monitor.update()

            batch_monitor.close()
            epoch_monitor.update()

        epoch_monitor.close()

        # After discriminator is trained on real images,
        # train discriminator on fake images and train generator to fool discriminator
        tqdm.write('Phase 2 (train discriminator/generator in a minimax game)')
        epoch_monitor = tqdm(total=self.num_epochs)
        epoch_monitor.set_description('Epoch')

        for epoch in range(self.num_epochs):
            batch_monitor = tqdm(total=len(self.data_loader))
            batch_monitor.set_description('Batch')

            for label, images0, images1 in self.data_loader:
                batch_size = images0.size(0)
                noise = to_variable(torch.randn(batch_size, self.noise_dim))

                images0 = to_variable(images0)
                images1 = to_variable(images1)
                label = to_variable(label)

                # Train discriminator to discriminate identity of real and fake images
                fake_images = self.generator(noise, images0)
                output1, output2 = self.discriminator(images1, fake_images)
                d_fake_loss = self.contrastive_loss(output1, output2, 0)

                # Backpropagation
                self.contrastive_loss.zero_grad()
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                d_fake_loss.backward()
                self.d_optimizer.step()

                # Train Generator to fool Discriminator
                noise = to_variable(torch.randn(batch_size, self.noise_dim))
                fake_images = self.generator(noise, images0)
                output1, output2 = self.discriminator(images1, fake_images)
                g_loss = self.contrastive_loss(output1, output2, 0, self.max_L2, images0,
                                               fake_images)
                batch_monitor.set_postfix(d_fake_loss=d_fake_loss.data[0], g_loss=g_loss.data[0])

                # Backpropagation
                self.contrastive_loss.zero_grad()
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                batch_monitor.update()

            batch_monitor.close()
            epoch_monitor.update()

        epoch_monitor.close()

        # Save generator to file
        g_path = os.path.join(self.model_path, 'generator-%d.pkl' % (epoch+1))
        torch.save(self.generator.state_dict(), g_path)

    def sample(self):
        """Sample images."""
        # Load trained parameters (generator)
        g_path = os.path.join(self.model_path, 'generator-%d.pkl' % (self.num_epochs))
        self.generator.load_state_dict(torch.load(g_path))
        self.generator.eval()

        # Sample the images
        i = 0
        for _, image, _ in self.data_loader:
            i = i+1
            noise = to_variable(torch.randn(1, self.noise_dim))
            fake_image = self.generator(noise, to_variable(image))
            fake_path = os.path.join(self.sample_path, 'fake'+str(i).zfill(4)+'.png')
            real_path = os.path.join(self.sample_path, 'real'+str(i).zfill(4)+'.png')
            torchvision.utils.save_image(denorm(to_variable(image).data), real_path, nrow=1)
            torchvision.utils.save_image(denorm(fake_image.data), fake_path, nrow=1)


def denorm(image):
    """Convert image range (-1, 1) to (0, 1)."""
    out = (image + 1) / 2
    return out.clamp(0, 1)


def to_variable(tensor):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)
