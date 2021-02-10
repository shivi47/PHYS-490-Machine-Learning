# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2021 (P Ronagh)
# Lecture 6--A Primer to ML R&D in PyTorch
#
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as func

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
        self.fc2= nn.Linear(100, 5)

    # Feedforward function
    def forward(self, x):
        h = func.relu(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

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

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()
