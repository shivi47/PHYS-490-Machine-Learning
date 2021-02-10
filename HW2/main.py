# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2021 (P Ronagh)
# Lecture 6--A Primer to ML R&D in PyTorch
#
# ------------------------------------------------------------------------------

import json, argparse, torch, sys, os
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sys.path.append('src')
from nn_gen import Net
from data_input import Data_Input


def plot_results(obj_vals, cross_vals):

    assert len(obj_vals)==len(cross_vals), 'Length mismatch between the curves'
    num_epochs= len(obj_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    plt.legend()
    plt.savefig(args.res_path + '/fig.pdf')
    plt.close()


def prep_demo(param):

    # Construct a model and dataset
    model= Net(param['n_bits'])
    data= Data_Input()
    return model, data


def run_demo(param, model, data):

    # Define an optimizer and the loss function
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss= torch.nn.BCELoss(reduction= 'mean')

    obj_vals= []
    cross_vals= []
    accuracy_train_vals = []
    accuracy_test_vals = []
    num_epochs= int(param['num_epochs'])


    # Training loop
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(data, loss, epoch, optimizer)
        obj_vals.append(train_val)

        test_val= model.test(data, loss, epoch)
        cross_vals.append(test_val)


        # High verbosity report in output stream
        if args.v>=2:
            if not ((epoch + 1) % param['display_epochs']):
                accuracy_train_val, accuracy_test_val = model.accuracy(data)
                accuracy_train_vals.append(accuracy_train_val)
                accuracy_test_vals.append(accuracy_test_val)
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(test_val)+\
                      '\tTraining Accuracy: {:.2f}%'.format(accuracy_train_val/len(data.y_train) * 100)+\
                      '\tTest Accuracy: {:.2f}%'.format(accuracy_test_val/len(data.y_test)*100))


    # Low verbosity final report
    if args.v:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))
        print('Final train accuracy: {:.4f}'.format(accuracy_train_vals[-1]))
        print('Final test accuracy: {:.4f}'.format(accuracy_test_vals[-1]))

    return obj_vals, cross_vals

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Tutorial')
    parser.add_argument('--param', metavar='param.json',
                        help='parameter file name')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    parser.add_argument('--res-path', metavar='results',
                        help='path of results')
    args = parser.parse_args()

    os.chdir('files/')
    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)


    model, data= prep_demo(param['data'])
    obj_vals, cross_vals= run_demo(param['exec'], model, data)
    #plot_results(obj_vals, cross_vals)
