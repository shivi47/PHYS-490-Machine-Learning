#!usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler

# TODO: main.py delegation; get both methods to run from a separate file and write in a 3rd file
# TODO: output file; get code to spit out an output file with the w vectors for both methods
# TODO: Analytic Solution; compute analytical solution for LRM just like gradient descent

#setting variables based in given arguments from command line
data = np.loadtxt('./%s' % sys.argv[1])
params = json.load(open('%s' % sys.argv[2]))


def GD(data, params):

    if len(data[0, :]) == 3:
    #input data
        x1 = data[:, 0]
        x2 = data[:, 1]
        y = data[:, 2]
        alpha = params['learning rate']
        epochs = params['num iter']  # number of iterations to perform gradient descent
        alpha = 0.0001
        epochs = 10000
    #performing Gradient Descent
        w = np.zeros([len(data[0, :]),1])
        print(w)
        phi = np.array([x1, x2])
        #print(np.dot(w.T, phi))
        for i in range(epochs):
            y_pred = np.dot(w[:-1, :].T, phi) + w[-1] # predicted value of y, updated for each iteration
            dL_dx1 = - np.sum((y - y_pred).dot(x1.T))  # derivative with respect to x1 feature
            dL_dx2 = - np.sum((y - y_pred).dot(x2.T))
            dL_dc = - np.sum(y - y_pred)
            w[0] -= alpha * dL_dx1
            w[1] -= alpha * dL_dx2
            w[-1] -= alpha * dL_dc
            #if i <= 10 or i >= 100 or i <= 200: print(w)
        w = np.round(w, 4)
        w = np.array([w[2], w[0], w[1]])
        print('\nw_gd1 = %s \nw_gd2 = %s \nw_gd3 = %s' % (w[0], w[1], w[2]))
        return w


    if len(data[0, :]) == 5:
    # input data
        x1 = data[:, 0]
        x2 = data[:, 1]
        x3 = data[:, 2]
        x4 = data[:, 3]
        x5 = np.ones([len(data[:, 0])])
        y = data[:, 4]
        alpha = params['learning rate']
        epochs = params['num iter']  # number of iterations to perform gradient descent
        alpha = 0.000001
        epochs = 100000
    # performing Gradient Descent
        w = np.zeros([len(data[0, :]), 1])
        print(w)
        phi = np.array([x1, x2, x3, x4, x5])
        for i in range(epochs):
            y_pred = np.dot(w.T, phi)  # predicted value of y, updated for each iteration
            dL_dx1 = - np.sum((y - y_pred).dot(x1.T))  # derivative with respect to x1 feature
            dL_dx2 = - np.sum((y - y_pred).dot(x2.T))
            dL_dx3 = - np.sum((y - y_pred).dot(x3.T))
            dL_dx4 = - np.sum((y - y_pred).dot(x4.T))
            dL_dx5 = - np.sum((y - y_pred).dot(x5.T))
            w[0] -= alpha * dL_dx1
            w[1] -= alpha * dL_dx2
            w[2] -= alpha * dL_dx3
            w[3] -= alpha * dL_dx4
            w[-1] -= alpha * dL_dx5
        w = np.round(w, 4)
        w = np.array([w[-1], w[0], w[1], w[2], w[3]])
        print('\nw_gd1 = %s \nw_gd2 = %s \nw_gd3 = %s \nw_gd4 = %s \nw_gd5 = %s' % (w[0], w[1], w[2], w[3], w[4]))
        return w


GD(data, params)