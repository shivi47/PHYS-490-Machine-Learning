import torch, json, argparse, os, sys
import torch.optim as optim
import matplotlib.pyplot as plt
sys.path.append('src')
from data_input import Data_Input
from nn_gen import Net


def prep(params, test_set):

    # Construct a model and dataset
    model= Net(params['n_bits'])
    data= Data_Input(test_set)
    return model, data

def run(params, model, data):

    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.paramaters(), lr=int(params['lr']))

    obj_vals= []
    cross_vals= []
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])

    # Training loop
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(optimizer, batch_size)
        obj_vals.append(train_val)

        test_val= model.test(test_set, batch_size)
        cross_vals.append(test_val)

        if args.v>=2:
            if not ((epoch + 1) % display_epochs):
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(test_val))

    if args.v:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    return obj_vals, cross_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Tutorial')
    parser.add_argument('--param', default='./data/params.json', metavar='./data/params.json',
                        help='parameter file name')
    parser.add_argument('-v', type=int, default=2, metavar='N',
                        help='verbosity (default: 1)')
    parser.add_argument('--o', '--result_dir', metavar='./results',
                        help='path of results')
    parser.add_argument('-n', '--num_images', type=int, default=100,
                        help='number of computer-written images to generate')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        params = json.load(paramfile)
    print("testing")
    test_set = int(params['test_set'])
    display_epochs = int(params['display_epochs'])

    results_dir = args.o

    model, data= prep(params, test_set)
    obj_vals, cross_vals= run(params, model, data)

    x_range = range(1, num_epochs + 1)
    plt.plot(x_range, obj_vals, label = 'Training Loss', color = 'red')
    plt.plot(x_range, cross_vals, label = 'Test Loss', color = 'blue')
    plt.title('BCE and KLD Loss: %i Iterations of Training' % num_epochs)
    plt.ylabel('Loss')
    plt.xlabel('Training Epochs')
    plt.legend()
    if args.o:
        if not os.path.exists(args.o):
            os.mkdir(args.o)
        plt.savefig(os.path.join(args.o, 'loss.pdf'))
    plt.show()

    model.make_images(args.n, args.o)
