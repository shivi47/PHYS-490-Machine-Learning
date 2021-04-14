import numpy as np
import json, os, sys, torch
from sklearn.model_selection import train_test_split

class Data_Input():
    def __init__(self, test_set):
        data_mnist = np.loadtxt('/Users/Shivani/Documents/Phys490/HW5/data/even_mnist.csv')

        x = data_mnist[:, :-1].astype(np.float)/255
        y = data_mnist[:, -1].astype(np.long)/2

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_set)

        x_train.reshape(-1,1,14,14)
        x_test.reshape(-1,1,14,14)

        self.x_train = torch.tensor(x_train, dtype=torch.float)
        self.x_test = torch.tensor(x_test, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        self.x_train = torch.tensor(x_train, dtype=torch.float)
        self.x_test = torch.tensor(x_test, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)


if __name__ == '__main__':
    Data_Input()
