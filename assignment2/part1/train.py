################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set
from tqdm.auto import tqdm
from copy import deepcopy


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights= 'ResNet18_Weights.DEFAULT')  
    for param in model.parameters():
        param.requires_grad = False

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = nn.Linear(in_features = model.fc.in_features, out_features= num_classes)  
    nn.init.normal_(model.fc.weight, mean = 0, std = .01)
    nn.init.constant_(model.fc.bias, 0.0) 

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir=data_dir, augmentation_name=augmentation_name)
    trainset = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers = 0, drop_last=True)
    valset = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=False)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    loss_module = nn.CrossEntropyLoss()
    val_accuracies = []
    model.to(device)

    for epoch in range(epochs):
        model.train()

        for image, label in tqdm(trainset):
            image, label = image.to(device), label.to(device)
            pred = model(image)
            loss = loss_module(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = evaluate_model(model=model, data_loader=valset, device=device)
        val_accuracies.append(accuracy)
        print(f'Current epoch: {epoch}')
        print('Validation accuracy:', accuracy)
        print('Loss:', loss.item()) 

        torch.cuda.empty_cache()
        
        if epoch == 0 or val_accuracies[epoch] > max(val_accuracies):
            bestmodel = deepcopy(model)

    # Load the best model on val accuracy and return it.
    model = bestmodel

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    true_preds, num_preds = 0., 0.

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds_softmax = F.softmax(preds, dim=1)
            _, pred_labels = torch.max(preds_softmax, 1)
            true_preds += (pred_labels == data_labels).sum().item()
            num_preds += data_labels.shape[0]

    accuracy = true_preds / num_preds
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 

    # Load the model
    model = get_model()

    # Get the augmentation to use
    pass

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, device=device, checkpoint_name= None, augmentation_name=augmentation_name) 

    # Evaluate the model on the test set
    test_dataset = get_test_set(data_dir=data_dir, test_noise=test_noise)
    testset = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=False)
    accuracy = evaluate_model(model, data_loader=testset, device=device) 
    print('Test accuracy:', accuracy)
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
