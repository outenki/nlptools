from torch import opti


def get_optimizer(net, algorithm, lr=0.001):
    if algorithm == 'rmsprop':
        optimizer = opti.RMSprop(net.parameters(), lr=lr)
    elif algorithm == 'sgd':
        optimizer = opti.SGD(net.parameters(), lr=lr)
    elif algorithm == 'adagrad':
        optimizer = opti.Adagrad(net.parameters(), lr=lr)
    elif algorithm == 'adadelta':
        optimizer = opti.Adadelta(net.parameters(), lr=lr)
    elif algorithm == 'adam':
        optimizer = opti.Adam(net.parameters(), lr=lr)
    elif algorithm == 'adamax':
        optimizer = opti.Adamax(net.parameters())
    return optimizer
