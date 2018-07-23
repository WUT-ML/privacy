"""Privacy-Preserving Representation-Learning Variational Generative Adversarial Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from reference_vgan_model import Generator, Discriminator
from reference_data_loader import FERGDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensorboardX as tb
import os

# Set parameters
image_size = 64
n_ids = 6
n_attrs = 7
n_epochs = 100
BATCH_SIZE = 256
MEAN = [0.2516, 0.1957, 0.1495]
STD = [0.2174, 0.1772, 0.1569]

# Define image transformation
dataset_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN,
                         std=STD)
])

# Prepare dataset loader
dataset = FERGDataset(transform=dataset_transform, path="../../../../FERG_DB_256/")
data_loader = DataLoader(dataset=dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=4,
                         shuffle=True,
                         drop_last=False)


def label_reshape_1d(label, label_dim):
    """Reshape label to list of zeros and ones."""
    out = torch.FloatTensor(len(label)).zero_()
    for i, l in enumerate(label):
        out[i] = l
    return out


def to_variable(tensor):
    """Convert tensor to variable."""
    return Variable(tensor.cuda())


def denorm(image):
    """Apply inverse transformation to noramlization."""
    std_tensor = torch.FloatTensor(STD)
    std_tensor.unsqueeze_(0)
    std_tensor = std_tensor.expand(BATCH_SIZE, -1)
    std_tensor.unsqueeze_(2)
    std_tensor.unsqueeze_(3)
    std_tensor = std_tensor.expand(-1, -1,  image_size, image_size)
    mean_tensor = torch.FloatTensor(MEAN)
    mean_tensor.unsqueeze_(0)
    mean_tensor = mean_tensor.expand(BATCH_SIZE, -1)
    mean_tensor.unsqueeze_(2)
    mean_tensor.unsqueeze_(3)
    mean_tensor - mean_tensor.expand(-1, -1, image_size, image_size)
    out = torch.add(torch.div(image, std_tensor.cuda()), mean_tensor.cuda())
    return out.clamp(0, 1)


# Initialize Generator and Discriminator networks
g = Generator()
d = Discriminator(n_ids, n_attrs)
g.cuda()
d.cuda()

# Define optimizers
g_optimizer = torch.optim.RMSprop(g.parameters(), 0.0002)
d_optimizer = torch.optim.RMSprop(d.parameters(), 0.0002)

# Tensorboar writer to visualize loss functions
tb_writer = tb.SummaryWriter()
step = 0

# Train generator GENERATOR_TRAIN_RATIO as aoften as discriminator
GENERATOR_TRAIN_RATIO = 1


# Learning process
for epoch in range(n_epochs):
    print("Epoch number: ", epoch)

    train_generator = GENERATOR_TRAIN_RATIO

    # Preapre a minibatch
    for images, id, attr in data_loader:

        # Reshape ids and attrs
        labels = label_reshape_1d(id, n_ids)
        attr = label_reshape_1d(attr, n_attrs)

        # Convert tensors to variables
        images = to_variable(images)
        labels = to_variable(labels)
        attr = to_variable(attr)

        # ----------------------TRAIN DISCRIMINATOR----------------------------
        if(train_generator == GENERATOR_TRAIN_RATIO):

            # Train Discriminator on real images
            output_fake, output_id, output_attr = d(images)

            # Train Discriminator to recognize real images as real
            loss = nn.BCEWithLogitsLoss().cuda()
            target = Variable(torch.ones(output_fake.size()), requires_grad=True).cuda()
            d_loss_real = loss(output_fake, target)

            # Train Discriminator to recognize id
            d_loss_id = F.cross_entropy(output_id, labels.long())

            # Train Discriminator to recognize attributes
            d_loss_attr = F.cross_entropy(output_attr, attr.long())

            # Train Discriminator to recognize fake images as fake
            batch_size = images.size(0)
            fake_ids = torch.LongTensor(batch_size, 1).random_() % n_ids
            fake_ids_onehot = torch.zeros(batch_size, n_ids)
            fake_ids_onehot.scatter_(1, fake_ids, 1)
            fake_ids_onehot = to_variable(fake_ids_onehot)

            fake_images, _, _ = g(images, fake_ids_onehot)
            output_fake, _, _ = d(fake_images)
            loss = nn.BCEWithLogitsLoss().cuda()
            target = Variable(torch.zeros(output_fake.size()), requires_grad=True).cuda()
            d_loss_fake = loss(output_fake, target)

            # Combine losses
            d_loss = 0.25 * d_loss_real + 0.5 * d_loss_id + 0.25 * d_loss_attr + 0.25 * d_loss_fake

            # Backpropagation for discriminator
            d.zero_grad()
            g.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Update tensorboard
            tb_writer.add_scalar('discriminator_loss', d_loss.data[0], step)
            tb_writer.add_scalar('d1_loss', d_loss_real.data[0], step)
            tb_writer.add_scalar('d1_loss_fake', d_loss_fake.data[0], step)
            tb_writer.add_scalar('d2_loss', d_loss_id.data[0], step)
            tb_writer.add_scalar('d3_loss', d_loss_attr.data[0], step)

            train_generator = 0

        else:
            train_generator += 1

        # ----------------------TRAIN GENERATOR----------------------------

        # Train Generator
        batch_size = images.size(0)
        fake_ids = torch.LongTensor(batch_size, 1).random_() % n_ids
        fake_ids_onehot = torch.zeros(batch_size, n_ids)
        fake_ids_onehot.scatter_(1, fake_ids, 1)
        fake_ids_onehot = to_variable(fake_ids_onehot)

        # Generate fake images with controlled ids
        fake_images, mu, logvar = g(images, fake_ids_onehot)
        output_fake, output_id, output_attr = d(fake_images)

        # Train Generator to fool fake/real Discriminator
        loss = nn.BCEWithLogitsLoss().cuda()
        target = Variable(torch.ones(output_fake.size()), requires_grad=True).cuda()
        g_loss_fake = loss(output_fake, target)

        # Train Generator to fool id Discriminator
        g_loss_id = F.cross_entropy(output_id, to_variable(fake_ids.long().squeeze()))

        # Train Generator to fool attr Discriminator
        g_loss_attr = F.cross_entropy(output_attr, attr.long())

        # Compute Kullback-Leibler divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Combine losses
        g_loss = 0.108 * g_loss_fake + 0.6 * g_loss_id + 0.29 * g_loss_attr + 0.002 * kld_loss

        # Backpropagation for generator
        g.zero_grad()
        d.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        tb_writer.add_scalar('generator_loss', g_loss.data[0], step)
        tb_writer.add_scalar('g1_loss', g_loss_fake.data[0], step)
        tb_writer.add_scalar('g2_loss', d_loss_id.data[0], step)
        tb_writer.add_scalar('g3_loss', d_loss_attr.data[0], step)
        tb_writer.add_scalar('kld_loss', kld_loss.data[0], step)

        step += 1

    # At the end of each tenth epoch save generator and discriminator to file
    if((epoch + 1) % 10 == 0):
        g_path = os.path.join(os.getcwd(), 'generator-%d.pkl' % (epoch+1))
        torch.save(g.state_dict(), g_path)
        d_path = os.path.join(os.getcwd(), 'discriminator-%d.pkl' % (epoch+1))
        torch.save(d.state_dict(), d_path)

    # At the end of each epoch generate sample images
    reals, fakes = [], []
    fake_ids_onehot = torch.zeros(BATCH_SIZE, n_ids)
    fake_ids_onehot = to_variable(fake_ids_onehot)

    for images, _, _ in data_loader:
        reals.append(denorm(to_variable(images).data))
        fakes.append(denorm((g(to_variable(images), fake_ids_onehot)[0]).data))
        break

    # Write images to tensorboard
    real_previews = torchvision.utils.make_grid(reals[0])
    fake_previews = torchvision.utils.make_grid(fakes[0])
    tb_writer.add_image('True images', real_previews, step)
    tb_writer.add_image('Generated images', fake_previews, step)

# Close tensorboard
tb_writer.close()
