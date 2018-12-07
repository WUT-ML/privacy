# coding=utf-8
"""
The purpose of this module is to perform transfer learning using pretraind ResNet model.

We start with ResNet without last fc layer, then train model (unfreeze all weights) with new data.
The train and validation images should be (by default) in ./data/train and ./data/val divided into
subdirectories: one subdirectory for one class.
At the end of each training epoch model is validated in order to improve classification accuracy.
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import argparse
import CelebA_dataset_generation


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs):
    """Train pretrained model using new data."""
    since = time.time()

    best_model_weights = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


def main():
    """Entry point for transfer learning."""
    percentage_of_validation = 10
    # If dataset is not yet present in the processed images path, it is now generated
    if not os.path.exists(config.image_processed_path):
        os.makedirs(config.image_processed_path)
        CelebA_dataset_generation.split_images_into_train_and_val(
            config.image_source_path, config.image_processed_path, config.CelebA_attribute,
            config.distance_weight, percentage_of_validation)

    # Transform data to get input consistent with ResNet model
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Set data directory
    data_dir = os.path.join(config.image_processed_path, 'attribute_' + str(
        config.CelebA_attribute), 'distortion_' + str(config.distance_weight))

    # Data should be split into two subdirs: 'train' and 'val'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    # Load a pretrained model and reset final fully connected layer.
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, len(class_names))
    model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.9)

    # Train and evaluate
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, num_epochs=config.num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--image_source_path', type=str)
    parser.add_argument('--image_processed_path', type=str)
    parser.add_argument('--CelebA_attribute', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--distance_weight', type=float)

    config = parser.parse_args()
    if config.dataset == 'CelebA' and (config.image_source_path is None):
        if config.distance_weight is None:
            config.image_source_path = os.path.join('data', 'CelebA')
            config.distance_weight = 0
        else:
            weight = str(config.distance_weight)
            config.image_source_path = os.path.join('results', 'CelebA', 'samples', weight)
    if config.dataset == 'CelebA' and (config.image_processed_path is None):
        config.image_processed_path = os.path.join('data', 'CelebA_for_utility')
        if not os.path.exists(config.image_processed_path):
            os.makedirs(config.image_processed_path)
    if config.dataset == 'FERG' and (config.num_epochs is None):
        print('FERG is not supported yet')
        exit(11)

    main()
