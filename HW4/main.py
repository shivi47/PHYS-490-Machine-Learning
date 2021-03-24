import numpy as np
import os, torch, json, argparse
import torch.nn as nn
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, metavar='data/in.txt',
                        default='data/in.txt',
                        help='Path to .txt file containing the dataset')
    parser.add_argument('--verbose', type=int, metavar='N', default=1,
                        help='verbosity, default at 1')
    args = parser.parse_args()

    data = np.loadtxt(args.data, dtype=str)
    verbosity = args.verbose
    batch = 100
    epochs = 50
    lr = 0.01
    w = np.ones([4])
    ising = np.empty((len(data),len(data[0])),dtype = np.float64)
    i = 0
    for row in data:
        j = 0
        for index in row:
            if index == '+':
                ising[i, j] = 1.0
            else: ising[i, j] = -1.0

            j += 1
        i += 1

    data = torch.from_numpy(ising)
    print(data)

    data = data.reshape(-1,4)
    print(len(data))
    data = np.array(data)
    data2 = [np.array_str(data[i,:]).replace('[','').replace(']','').replace('\n','') for i in range(0,len(data))]
    frequency_in = Counter(data2)
    for key in frequency_in.keys():
        value = frequency_in[key] / len(data2)
        p_in = {key : value}
    p = torch.from_numpy(np.array([p_in[key] for key in p_in.keys()]))
    e = np.sum([data[:,i]*data[:,(i+1)%len(data[0])] for i in range(len(data[0]))], axis=1)/len(data)
    training = []

    for epoch in range(1, epochs+1):
        X = []
        for i in range(0, batch):
            x1 = np.random.choice([-1, 1], len(data[0]))
            x2  = np.random.choice([-1, 1], len(data[0]))

            for j in range(len(data[0])):
                list(x1)
                list(x2)
                for k in range(len(data[0])):
                    y1 = np.exp(-w*np.sum([-x1[k]*x1[(k+1)%len(data[0])]]))
                    y2 = np.exp(-w*np.sum([-x2[k]*x2[(k+1)%len(data[0])]]))

                    if np.random.rand() < y2[j]/y1[j]:
                        x1[j] = x2[j]
                    else: continue

            x1 = list(x1)
            X.append(x1)
        X = np.array(X)
        X2 = [np.array_str(X[i,:]).replace('[','').replace(']','').replace('\n','') for i in range(len(X))]
        frequency_model = Counter(X2)
        for key in frequency_model.keys():
            value = frequency_model[key] / len(X2)
            p_mod = {key : value}
            if len(p_mod.keys()) < len(p_in.keys()):
                for key in p_in.keys():
                    try:
                        p_mod[key]
                    except:
                        p_mod[key] = 1e-8
        p_model = torch.from_numpy(np.array([p_mod[key] for key in p_in.keys()]))
        for i in range(len(p)):
            training.append(np.sum(p[i] * np.log(p[i]/p_model[i])))
        e_model = np.sum([X[:,i]*X[:,(i+1)%len(X[0])] for i in range(len(X[0]))], axis=1)/len(data)

        w = w + lr*(e - e_model)

        if verbosity > 1 and epoch%n_epoch_v == 0:
            print('[{}/{}] weights: {}; KL Divergence Loss: {:4f}'.format(epoch, epochs, w, training[-1]))

        for i in range(len(data[0])):
            print('({},{}):{}'.format(i,i+1,int(round(w[i]))))

            print('{(0,1) : %i, (1,2) : %i, (2,3) : %i, (3,0) : %i}' % (round(w[0]),
                                                                           round(w[1]),
                                                                           round(w[2]),
                                                                           round(w[3])))
