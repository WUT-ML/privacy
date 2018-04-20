"""Privacy-Preserving Representation-Learning Variational Generative Adversarial Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from reference_vgan_model import Generator, Discriminator
from reference_data_loader import FERGDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensorboardX as tb

# Set parameters
image_size = 64
n_ids = 6
n_attrs = 7
n_epochs = 200

# Define image transformation
dataset_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=image_size),
    transforms.ToTensor(),
])

# Prepare dataset loader
dataset = FERGDataset(transform=dataset_transform, path="../../../../../../FERG_DB_256/")
data_loader = DataLoader(dataset=dataset,
                         batch_size=256,
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

# Learning process
for epoch in range(n_epochs):
    print("Epoch number: ", epoch)
    

    train_discriminator = True

    # Preapre a minibatch
    for images, id, attr in data_loader:

        # Reshape ids and attrs
        labels = label_reshape_1d(id, n_ids)
        attr = label_reshape_1d(attr, n_attrs)

        # Convert tensors to variables
        images = to_variable(images)
        labels = to_variable(labels)
        attr = to_variable(attr)

        #----------------------TRAIN DISCRIMINATOR----------------------------
        if(train_discriminator):

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
        
            fake_images = g(images, fake_ids_onehot)
            output_fake, _, _ = d(fake_images)
            loss = nn.BCEWithLogitsLoss().cuda()
            target = Variable(torch.zeros(output_fake.size()), requires_grad=True).cuda()
            d_loss_fake = loss(output_fake, target)

            # Combine losses
            d_loss = 0.25 * d_loss_real + 0.5 * d_loss_id  + 0.25 * d_loss_attr + 0.25 * d_loss_fake
        
            # Backpropagation for discriminator
            d.zero_grad()
            g.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Update tensorboard
            tb_writer.add_scalar('discriminator_loss', d_loss.data[0], step)

            # Train generator twice as often as discriminator
            train_discriminator = not train_discriminator



        #----------------------TRAIN GENERATOR----------------------------

        # Train Generator
        batch_size = images.size(0)
        fake_ids = torch.LongTensor(batch_size, 1).random_() % n_ids
        fake_ids_onehot = torch.zeros(batch_size, n_ids)
        fake_ids_onehot.scatter_(1, fake_ids, 1)
        fake_ids_onehot = to_variable(fake_ids_onehot)

        # Generate fake images with controlled ids
        fake_images = g(images, fake_ids_onehot)
        output_fake, output_id, output_attr = d(fake_images)

        # Train Generator to fool fake/real Discriminator 
        loss = nn.BCEWithLogitsLoss().cuda()
        target = Variable(torch.ones(output_fake.size()), requires_grad=True).cuda()
        g_loss_fake = loss(output_fake, target)
        
        # Train Generator to fool id Discriminator
        g_loss_id = F.cross_entropy(output_id, to_variable(fake_ids.long().squeeze()))

        # Train Generator to fool attr Discriminator
        g_loss_attr = F.cross_entropy(output_attr, attr.long())

        # Combine losses 
        g_loss = 0.11 * g_loss_fake + 0.6 * g_loss_id  + 0.29 * g_loss_attr

        # Backpropagation for generator
        g.zero_grad()
        d.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        tb_writer.add_scalar('generator_loss', g_loss.data[0], step)
        step += 1

    # At the end of each tenth epoch save generator and discriminator to file
    if( ( epoch + 1 ) % 10 == 0):
        g_path = os.path.join(os.getcwd(), 'generator-%d.pkl' % (epoch+1))
        torch.save(g.state_dict(), g_path)
        d_path = os.path.join(os.getcwd(), 'discriminator-%d.pkl' % (epoch+1))
        torch.save(d.state_dict(), d_path)

# Close tensorboard
tb_writer.close()
