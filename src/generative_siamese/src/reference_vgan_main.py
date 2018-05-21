"""Privacy-Preserving Representation-Learning Variational Generative Adversarial Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from reference_vgan_model import Generator, Discriminator, SiameseDiscriminator, DistanceBasedLoss
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
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
SIAM_FACTOR = 0

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
    image = (image * 0.5) + 0.5
    return image.clamp(0, 1)


# Initialize Generator and Discriminator networks
g = Generator()
d = Discriminator(n_ids, n_attrs)
siam = SiameseDiscriminator(image_size)
distance_loss = DistanceBasedLoss(2.0)
g.cuda()
d.cuda()
siam.cuda()
distance_loss.cuda()

# Define optimizers
g_optimizer = torch.optim.RMSprop(g.parameters(), 0.0003)
d_optimizer = torch.optim.RMSprop(d.parameters(), 0.0003)
siam_optimizer = torch.optim.RMSprop(siam.parameters(), 0.0003)

# Tensorboar writer to visualize loss functions
tb_writer = tb.SummaryWriter()
step = 0

# Train generator GENERATOR_TRAIN_RATIO as often as discriminator
GENERATOR_TRAIN_RATIO = 2


# Learning process
for epoch in range(n_epochs):
    print("Epoch number: ", epoch)

    train_generator = GENERATOR_TRAIN_RATIO

    # Preapre a minibatch
    for images0, id, attr, images1, labels_binary in data_loader:

        # Reshape ids and attrs
        labels = label_reshape_1d(id, n_ids)
        attr = label_reshape_1d(attr, n_attrs)

        # Convert tensors to variables
        images0 = to_variable(images0)
        labels = to_variable(labels)
        attr = to_variable(attr)

        images1 = to_variable(images1)
        labels_binary = to_variable(labels_binary)

        # ----------------------TRAIN DISCRIMINATOR----------------------------
        if(train_generator == GENERATOR_TRAIN_RATIO):

            # Train Discriminator on real images
            output_fake, output_id, output_attr = d(images0)

            # Train Discriminator to recognize real images as real
            loss = nn.BCELoss().cuda()
            target = Variable(torch.ones(output_fake.size()), requires_grad=False).cuda()
            d_loss_real = loss(output_fake, target)

            # Train Discriminator to recognize id
            d_loss_id = F.cross_entropy(output_id, labels.long())

            # Train Discriminator to recognize attributes
            d_loss_attr = F.cross_entropy(output_attr, attr.long())
            #attr_accuracy = (output_attr.max(1)[1]-attr.long()).nonzero().size(0)/BATCH_SIZE
            #tb_writer.add_scalar('discriminator/attr_accuracy', attr_accuracy, step)

            # Train Discriminator to recognize fake images as fake
            batch_size = images0.size(0)
            fake_ids = torch.LongTensor(batch_size, 1).random_() % n_ids
            fake_ids_onehot = torch.zeros(batch_size, n_ids)
            fake_ids_onehot.scatter_(1, fake_ids, 1)
            fake_ids_onehot = to_variable(fake_ids_onehot)

            fake_images, _, _ = g(images0, fake_ids_onehot)
            output_fake, _, _ = d(fake_images)
            loss = nn.BCELoss().cuda()
            target = Variable(torch.zeros(output_fake.size()), requires_grad=False).cuda()
            d_loss_fake = loss(output_fake, target)

            # Train Siamese Discriminator on real images
            d_siam_real = distance_loss(*siam(images0, images1), labels_binary)

            # Train Siamese Discriminator on fake images
            d_siam_fake = distance_loss(*siam(fake_images, images1), to_variable(torch.zeros(batch_size)))

            # Combine losses
            d_loss = (1.0 - SIAM_FACTOR/100.0) * (0.25 * d_loss_real + 0.5 * d_loss_id + 0.25 * d_loss_attr + 0.25 * d_loss_fake) + (SIAM_FACTOR/100.0) * (0.5 * d_siam_real + 0.5 * d_siam_fake)

            # Backpropagation for discriminator
            d.zero_grad()
            g.zero_grad()
            siam.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            siam_optimizer.step()

            # Update tensorboard
            tb_writer.add_scalar('discriminator/discriminator_loss', d_loss.data[0], step)
            tb_writer.add_scalar('discriminator/d1_loss', d_loss_real.data[0], step)
            tb_writer.add_scalar('discriminator/d1_loss_fake', d_loss_fake.data[0], step)
            tb_writer.add_scalar('discriminator/d2_loss', d_loss_id.data[0], step)
            tb_writer.add_scalar('discriminator/d3_loss', d_loss_attr.data[0], step)
            tb_writer.add_scalar('discriminator/d_siam_real', d_siam_real.data[0], step)
            tb_writer.add_scalar('discriminator/d_siam_fake', d_siam_fake.data[0], step)

            train_generator = 0

        else:
            train_generator += 1

        # ----------------------TRAIN GENERATOR----------------------------

        # Train Generator
        batch_size = images0.size(0)
        fake_ids = torch.LongTensor(batch_size, 1).random_() % n_ids
        fake_ids_onehot = torch.zeros(batch_size, n_ids)
        fake_ids_onehot.scatter_(1, fake_ids, 1)
        fake_ids_onehot = to_variable(fake_ids_onehot)

        # Generate fake images with controlled ids
        fake_images, mu, logvar = g(images0, fake_ids_onehot)
        output_fake, output_id, output_attr = d(fake_images)

        # Train Generator to fool fake/real Discriminator
        loss = nn.BCELoss().cuda()
        target = Variable(torch.ones(output_fake.size()), requires_grad=False).cuda()
        g_loss_fake = loss(output_fake, target)

        # Train Generator to fool id Discriminator
        g_loss_id = F.cross_entropy(output_id, to_variable(fake_ids.long().squeeze()))

        # Train Generator to fool attr Discriminator
        g_loss_attr = F.cross_entropy(output_attr, attr.long())

        # Compute Kullback-Leibler divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Train Siamese Discriminator
        g_siam = distance_loss(*siam(fake_images, images1), to_variable(torch.ones(batch_size)))

        # Combine losses
        g_loss = (1.0 - SIAM_FACTOR/100.0) * (0.108 * g_loss_fake + 0.6 * g_loss_id + 0.29 * g_loss_attr + 0.002 * kld_loss) + (SIAM_FACTOR/100.0) * g_siam

        # Backpropagation for generator
        g.zero_grad()
        d.zero_grad()
        siam.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        tb_writer.add_scalar('generator/generator_loss', g_loss.data[0], step)
        tb_writer.add_scalar('generator/g1_loss', g_loss_fake.data[0], step)
        tb_writer.add_scalar('generator/g2_loss', d_loss_id.data[0], step)
        tb_writer.add_scalar('generator/g3_loss', d_loss_attr.data[0], step)
        tb_writer.add_scalar('generator/kld_loss', kld_loss.data[0], step)
        tb_writer.add_scalar('generator/g_siam', g_siam.data[0], step)

        step += 1

    # At the end of each tenth epoch save generator and discriminator to file
    if((epoch + 1) % 10 == 0):
        g_path = os.path.join(os.getcwd(), 'generator-%d-%d.pkl' % (epoch+1, SIAM_FACTOR))
        torch.save(g.state_dict(), g_path)
        d_path = os.path.join(os.getcwd(), 'discriminator-%d-%d.pkl' % (epoch+1, SIAM_FACTOR))
        torch.save(d.state_dict(), d_path)

    # At the end of each epoch generate sample images
    reals, fakes = [], []
    fake_ids = torch.LongTensor(BATCH_SIZE, 1).random_() % n_ids
    fake_ids_onehot = torch.zeros(BATCH_SIZE, n_ids)
    fake_ids_onehot.scatter_(1, fake_ids, 1)
    fake_ids_onehot = to_variable(fake_ids_onehot)

    for images, _, _, _, _ in data_loader:
        reals.append(denorm(to_variable(images).data))
        fakes.append(denorm((g(to_variable(images), fake_ids_onehot)[0]).data))
        break

    # Write images to tensorboard
    real_previews = torchvision.utils.make_grid(reals[0])
    fake_previews = torchvision.utils.make_grid(fakes[0])
    tb_writer.add_image('GAN/True images', real_previews, step)
    tb_writer.add_image('GAN/Generated images', fake_previews, step)

# Close tensorboard
tb_writer.close()
