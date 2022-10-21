import numpy as np
import torch
import torch.nn.functional as F

import scipy.ndimage


# The eddy viscosity is
# mu_scale*Int[g(x - y; sigma) NN(y)]dy
def net_eval(net, x, mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):
    mu = net(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten()
    # data (prediction) clean

    if non_negative:
        mu[mu <= 0.0] = 0.0

    if filter_on:
        # the axis is 1
        n_f = len(x) // n_data
        for i in range(n_data):
            mu[i * n_f:(i + 1) * n_f] = scipy.ndimage.gaussian_filter1d(mu[i * n_f:(i + 1) * n_f], filter_sigma,
                                                                        mode="nearest")

    return mu * mu_scale


# x is nx by n_feature matrix, which is the input for the neural network
def nn_viscosity(net, x, mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):
    mu = net_eval(x=x, net=net, mu_scale=mu_scale, non_negative=non_negative, filter_on=filter_on,
                  filter_sigma=filter_sigma, n_data=n_data)
    return mu


# x is nx by n_feature matrix, which is the input for the neural network
def fno_viscosity(net, x, mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):
    mu = net_eval(x=x[np.newaxis, ...], net=net, mu_scale=mu_scale, non_negative=non_negative, filter_on=filter_on,
                  filter_sigma=filter_sigma, n_data=n_data)
    return mu


def get_act(activation):
    if activation == 'tanh':
        func = F.tanh
    elif activation == 'gelu':
        func = F.gelu
    elif activation == 'relu':
        func = F.relu_
    elif activation == 'elu':
        func = F.elu_
    elif activation == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{activation} is not supported')
    return func


def count_params(model):
    num = 0
    for p in model.parameters():
        num += p.numel()
    return num