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
from matplotlib import pyplot as plt, cm as cm


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
        self.fc2= nn.Linear(100, 10)
        self.fc3= nn.Linear(100, 10)

    # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=5),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=10),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

    # Decoder
    self.decoder = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 14*14),
            nn.Sigmoid()
        )

    def encoding(self, h):
        h = self.encoder(h)
        h = func.relu(self.fc1(h))
        u = self.fc2(h)
        var = self.fc3(h)
        stdev = torch.exp(0.5*lvar)
        e = torch.randn_like(stdev)
        return u, lvar

    def repar(self, u, var):
        stdev = torch.exp(0.5*lvar)
        e = torch.randn_like(stdev)
        return u + e * stdev

    def decoding(self, y):
        y = self.decoder(y)
        return y

    def FF(self, h):
        u, lvar = self.encoding(h)
        y = self.repar(u, lvar)
        return self.decoding(y), u, lvar

    def loss(self, rec, h, u, lvar):
        BCE = func.binary_cross_entropy(rec, h.view(-1, 14*14), reduction='sum')
        var = np.exp(lvar)
        KLD = 0.5 * torch.sum(1 + lvar - var - u**2)
        return(BCE - KLD)h.size(0)

    # Backpropagation function
    def backprop(self, optimizer, batch_size):
        self.train()
        optimizer.zero_grad()
        inputs= torch.from_numpy(data.x_train)
        batching = randint(0, len(inputs) - batch_size)
        recon, u, var = self(inputs[batching: batching + batch_size])
        obj_val = self.loss(recon,
                            self(inputs[batching: batching + batch_size]),
                            u,
                            lvar)
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
    def test(self, test_set):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            batching = randint(0, len(inputs) - test_set)
            recon, u, var = self(inputs[batching: batching + test_size])
            cross_val = self.loss(recon,
                                self(inputs[batching: batching + test_size]),
                                u,
                                lvar)
        return cross_val.item()

    def make_images(self, n, result_dir):
        with torch.no_grad():
            for i in range(0, n):
                test = torch.randn(1, 10)
                recon = self.decoding(test)
                image = recon.view(-1,14,14).numpy()
                fig, ax = plt.subplots()
                ax.imshow(recon[i, :, :], cmap = cm.binary)
                plt.axis('off')
                if result_dir:
                    plt.savefig(os.path.join(result_dir, '{}.pdf'.format(i))
            plt.show()
