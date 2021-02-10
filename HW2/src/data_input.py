import numpy as np
import json, os, sys
import torch

class Data_Input():
    def __init__(self):
        data_mnist = np.loadtxt('/Users/Shivani/Documents/Phys490/HW2/even_mnist.csv')
        y = data_mnist[:, -1]
        y = list(y)
        for i in range(len(y)):
            if y[i] == 0: y[i] = [1,0,0,0,0]
            if y[i] == 2: y[i] = [0,1,0,0,0]
            if y[i] == 4: y[i] = [0,0,1,0,0]
            if y[i] == 6: y[i] = [0,0,0,1,0]
            if y[i] == 8: y[i] = [0,0,0,0,1]
        y = np.array(y)

        x_train= data_mnist[:-3000, :-1]
        y_train= y[:-3000, :]
        x_train= np.array(x_train, dtype= np.float32)
        y_train= np.array(y_train, dtype= np.float32)
        print(x_train.shape, x_train.dtype, y_train.shape, y_train.dtype)
        #print(y_train)

        x_test= data_mnist[-3000:, :-1]
        y_test= y[-3000:, :]
        x_test= np.array(x_test, dtype= np.float32)
        y_test= np.array(y_test, dtype= np.float32)
        print(x_test.shape, x_test.dtype, y_test.shape, y_test.dtype)


        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test

if __name__ == '__main__':
    Data_Input()
