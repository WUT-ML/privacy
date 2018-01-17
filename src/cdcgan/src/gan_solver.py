"""Deep convolutional GAN solver."""
import os
import torch
from torch.autograd import Variable
import torchvision
from gan_model import Discriminator, Generator
from data_loader import label_reshape_1d


class GanSolver(object):
    """Solving GAN neural network."""

    def __init__(self, config, data_loader):
        """Set parameters of GAN neural network and its training."""
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_conv_dim = 128
        self.d_conv_dim = 128
        self.noise_dim = 100
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.learning_rate = 0.0001
        self.image_size = config.image_size
        self.num_epochs = config.num_epochs
        self.sample_size = 1
        self.number_of_samples = config.number_of_samples
        self.number_of_classes = 10
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.data_loader = data_loader
        self.save_only_last_model = config.save_only_last_model
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        self.generator = Generator(self.image_size, self.g_conv_dim,
                                   self.noise_dim + self.number_of_classes)
        self.discriminator = Discriminator(self.image_size, self.d_conv_dim,
                                           1 + self.number_of_classes)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), self.learning_rate, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), self.learning_rate, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

    def train(self):
        """Train generator and discriminator."""
        for epoch in range(self.num_epochs):
            for images, y_reshaped_1d, y_reshaped_2d in self.data_loader:
                # Train Discriminator
                images = to_variable(images)
                y_reshaped_1d = to_variable(y_reshaped_1d)
                y_reshaped_2d = to_variable(y_reshaped_2d)
                batch_size = images.size(0)
                noise = to_variable(torch.randn(batch_size, self.noise_dim))

                # Train Discriminator to recognize real images as real
                outputs = self.discriminator(images, y_reshaped_2d)
                real_loss = torch.mean((outputs - 1) ** 2)

                # Train Discriminator to recognize fake images as fake
                fake_images = self.generator(noise, y_reshaped_1d)
                outputs = self.discriminator(fake_images, y_reshaped_2d)
                fake_loss = torch.mean(outputs ** 2)

                # Backpropagation
                d_loss = real_loss + fake_loss
                self.discriminator.zero_grad()
                self.generator.zero_grad()

                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                noise = to_variable(torch.randn(batch_size, self.noise_dim))

                # Train Generator so that Discriminator recognizes G(z) as real.
                fake_images = self.generator(noise, y_reshaped_1d)
                outputs = self.discriminator(fake_images, y_reshaped_2d)
                g_loss = torch.mean((outputs - 1) ** 2)

                # Backpropagation
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # Print log after each epoch
            print(
                'Epoch [%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f'
                % (
                    epoch+1, self.num_epochs, real_loss.data[0], fake_loss.data[0],
                    g_loss.data[0]))

            if (not self.save_only_last_model or (epoch + 1) == self.num_epochs):
                # Save the model parameters for each (or only last) epoch
                g_path = os.path.join(self.model_path, 'generator-%d.pkl' % (epoch+1))
                d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' % (epoch+1))
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)

    def sample(self):
        """Sample images."""
        try:
            # Load trained parameters
            g_path = os.path.join(self.model_path, 'generator-%d.pkl' % (self.num_epochs))
            d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' % (self.num_epochs))
            self.generator.load_state_dict(torch.load(g_path))
            self.discriminator.load_state_dict(torch.load(d_path))
            self.generator.eval()
            self.discriminator.eval()

            # Sample the images, for each class self.number_of_samples will be outputed
            for c in range(0, self.number_of_classes):
                for i in range(0, self.number_of_samples):
                    noise = to_variable(torch.randn(self.sample_size, self.noise_dim))
                    y_reshaped_1d = label_reshape_1d(c, self.number_of_classes)
                    y_reshaped_1d = torch.unsqueeze(y_reshaped_1d, 0)
                    y_reshaped_1d = torch.cat(([y_reshaped_1d] * self.sample_size), 0)
                    y_reshaped_1d = to_variable(y_reshaped_1d)

                    fake_images = self.generator(noise, y_reshaped_1d)
                    sample_path = os.path.join(self.sample_path,
                                               'fake_sample-{i:08d}-c{c:04d}.png'.format(i=i, c=c))
                    torchvision.utils.save_image(denorm(fake_images.data), sample_path, nrow=1)

                    print("Saved sampled images to '%s'" % sample_path)

        except FileNotFoundError as err:
            print("Error: "+str(err))


def to_variable(tensor):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def denorm(image):
    """Convert image range (-1, 1) to (0, 1)."""
    out = (image + 1) / 2
    return out.clamp(0, 1)
