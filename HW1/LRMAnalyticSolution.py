#!usr/bin/python3

import sys
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
import json


#setting variables based in given arguments from command line
data = np.loadtxt('./%s' % sys.argv[1])
params = json.load(open('%s' % sys.argv[2]))


def AS(data, params):
#input data
    if len(data[0, :]) == 3:
        x1 = data[:, 0]
        x2 = data[:, 1]
        alpha = params['learning rate']
        epochs = params['num iter']  # number of iterations to perform gradient descent
        T = data[:, -1:]
#performing Analytical Solution
        Phi = np.array([np.ones([len(T)]), x1, x2]).T
        print(Phi.T)
        wOptimal = inv(dot(Phi.T, Phi)).dot(Phi.T).dot(T)
        wOptimal = np.round(wOptimal, 4)
        print('\nw_gd1 = %s \nw_gd2 = %s \nw_gd3 = %s' % (wOptimal[0], wOptimal[1], wOptimal[2]))
        return wOptimal

    elif len(data[0, :]) == 5:
        x1 = data[:, 0]
        x2 = data[:, 1]
        x3 = data[:, 2]
        x4 = data[:, 3]
        alpha = params['learning rate']
        epochs = params['num iter']  # number of iterations to perform gradient descent
        T = data[:, -1:]
# performing Analytical Solution
        Phi = np.array([np.ones([len(T)]), x1, x2, x3, x4]).T
        print(Phi.T)
        wOptimal = inv(dot(Phi.T, Phi)).dot(Phi.T).dot(T)
        print('\nw_gd1 = %s \nw_gd2 = %s \nw_gd3 = %s \nw_gd4 = %s \nw_gd5 = %s' % (wOptimal[0], wOptimal[1], wOptimal[2], wOptimal[3], wOptimal[4]))
        return wOptimal

AS(data, params)