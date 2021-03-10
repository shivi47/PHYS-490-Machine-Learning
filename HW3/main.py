import numpy as np
import matplotlib.pyplot as plt
import math, torch
from torch import nn
from torch.autograd import Variable


def rand_points(n_points= 1, lb= -1, ub= 1):
    return np.random.uniform(lb, ub, n_points)


def create_dataset(fx, fy, dataset_size=200, look_back=1, epsilon=0.01):
    random_x = rand_points(dataset_size)
    random_y = rand_points(dataset_size)

    data_in, data_out = [], []
    for x, y in zip(random_x, random_y):
        points = [(x + epsilon * fx(x, y), y + epsilon * fy(x, y))]

        for i in range(look_back):
            x1, y1 = points[-1]
            points.append((x1 + epsilon * u(x1, y1), y1 + epsilon * v(x1, y1)))

        data_in.append(points[:-1])
        data_out.append(points[-1])

    return np.array(data_in), np.array(data_out)



class lstm_reg(nn.Module):
    def __init__(self, n_dim, seq_len, n_hidden, n_layers=1):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(n_dim, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden * seq_len, n_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        b, s, h = x.shape
        x = self.fc(x)
        return x

"""
seq_len= 10
dataset_size= 10000
lb, ub= -1, 1

u = lambda x, y: -y/np.sqrt(x**2 + y**2)
v = lambda x, y: x/np.sqrt(x**2 + y**2)

# u = lambda x, y: np.sin(2*x) + np.sin(2*y)
# v = lambda x, y: np.cos(2*y)

x, y = np.meshgrid(np.linspace(lb, ub, 10), np.linspace(lb, ub, 10))
plt.quiver(x, y, u(x, y), v(x, y))
plt.show()


data_in, data_out = create_dataset(u, v, dataset_size, seq_len, 0.01)
train_in = torch.from_numpy(data_in.reshape(-1, seq_len, 2))
train_out = torch.from_numpy(data_out.reshape(-1, 1, 2))

net = lstm_reg(2, 1, 20)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2)

for e in range(1000):
    var_in = Variable(train_in).to(torch.float32)
    var_out = Variable(train_out).to(torch.float32)

    out = net(var_in)
    loss = criterion(out, var_out)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0:
        print(loss)

net = net.eval()
n_tests = 5

x, y = np.meshgrid(np.linspace(lb, ub, 10), np.linspace(lb, ub, 10))
plt.quiver(x, y, u(x, y), v(x, y))

for j in range(n_tests):
    x, y = rand_points(), rand_points()
    init_point = torch.from_numpy(np.array((x, y)).reshape(1, 1, 2))
    init_point = Variable(init_point).to(torch.float32)

    all_points = []
    for i in range(10000):
        init_point = net(init_point)
        all_points.append(init_point.detach().numpy())

    all_points = np.array(all_points).reshape(-1, 2)
    plt.plot(*np.array(all_points).T)
    plt.plot(x, y, 'o', markersize=5, color=plt.gca().lines[-1].get_color())

plt.show()
"""

if '__name__' == '__main__':

    parser = argparse.ArgumentParser(description='HW3 parser help')
    parser.add_argument('--param', metavar='param.json',
                        help='parameter file name')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    parser.add_argument('--res-path', metavar='results',
                        help='path of results')
    parser.add_argument('--x-field', default=-y/np.sqrt(x**2 + y**2), metavar='x**2',
                        help='expression of the x-component of the  vector field')
    parser.add_argument('--y-field', default=x/np.sqrt(x**2 + y**2), metavar='y**2',
                        help='expression of the y-component of the vector field')
    parser.add_argument('--lb', default=-1.0, metavar='LB',
                        help='lower bound for initial conditions (default: -1)')
    parser.add_argument('--ub', default=1.0, metavar='UB',
                        help='upper bound for initial conditions (default: 1)')
    parser.add_argument('--n-tests', default=3, metavar='N_TESTS',
                        help='number of test trajectories to plot')
    parser.add_argument('--help', help='usage: main.py [-h] [--param param.json] [-v N] [--res-path results] [--x-field x**2] [--y-field y**2] [--lb LB] [--ub UB] [--n-tests N_TESTS]')
    args = parser.parse_args()

    """
    os.chdir('param/')
    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    """

    lb = args.lb
    ub = args.ub
    xfield = args.xfield
    yfield = args.yfield
    ntests = args.ntests

    seq_len= 10
    dataset_size= 10000
    ntests= 3

    u = lambda x, y: -y/np.sqrt(x**2 + y**2)
    v = lambda x, y: x/np.sqrt(x**2 + y**2)

    x, y = np.meshgrid(np.linspace(lb, ub, 10), np.linspace(lb, ub, 10))
    plt.quiver(x, y, u(x, y), v(x, y))
    plt.show()

    data_in, data_out = create_dataset(u, v, dataset_size, seq_len, 0.01)

    for j in range(ntests):
        x, y = rand_points(), rand_points()
        print(x, y)
