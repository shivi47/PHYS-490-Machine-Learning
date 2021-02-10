# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2021 (P Ronagh)
# Lecture 6--A Primer to ML R&D in PyTorch
#
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions relu and sigmoid.
    '''

    def __init__(self, n_bits):
        super(Net, self).__init__()
        self.fc1= nn.Linear(n_bits, 100)
        self.fc2= nn.Linear(100, 100)
        self.fc3= nn.Linear(100, 100)
        self.fc4= nn.Linear(100, 5)

    # Feedforward function
    def forward(self, x):
        h1 = func.relu(self.fc1(x))
        h2 = func.relu(self.fc2(h1))
        h3 = func.relu(self.fc3(h2))
        y = torch.sigmoid(self.fc4(h3))
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)
        outputs= self(inputs)
        # An alternative to what you saw in the jupyter notebook is to
        # flatten the output tensor. This way both the targets and the model
        # outputs will become 1-dim tensors.
        obj_val= loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    def accuracy(self, data):
        accurate_train = 0
        accurate_test = 0
        with torch.no_grad():
            target_train = self.forward(torch.from_numpy(data.x_train))
            target_test = self.forward(torch.from_numpy(data.x_test))

            for i in range(len(data.y_train)):
                #print(target_train[i], data.y_train[i])
                if np.allclose(target_train.numpy()[i], data.y_train[i], atol = 1e-1): accurate_train += 1
            for i in range(len(data.y_test)):
                #print(target_train[i], y_train[i])
                if np.allclose(target_test.numpy()[i], data.y_test[i], atol = 1e-2): accurate_test += 1

        return accurate_train, accurate_test


    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()
