import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        im_channels, im_height, im_width = im_size
        self.conv = nn.Conv2d(im_channels, hidden_dim, kernel_size)
        width_hidden = im_width + 1 - kernel_size
        height_hidden = im_height +1 -kernel_size
        self.layer1 = nn.Linear(width_hidden*height_hidden*hidden_dim, n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        N, C, H, W = images.shape
        hidden_flat = F.relu(self.conv(images)).view(N, -1)
        scores = self.layer1(hidden_flat)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

