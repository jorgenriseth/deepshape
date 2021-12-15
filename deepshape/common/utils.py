import numpy as np
import torch
import pickle


def numpy_nans(dim, *args, **kwargs):
    arr = np.empty(dim, *args, **kwargs)
    arr.fill(np.nan)
    return arr


def col_linspace(start, end, N):
    return torch.linspace(start, end, N).unsqueeze(-1)


def torch_square_grid(k=64):
    Y, X = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X = torch.stack((X, Y), dim=-1)
    return X


def component_mse(inputs, targets, component: int):
    """ Stored here for now, will probably be moved elsewhere in the future"""
    return torch.sum((inputs[..., component] - targets[..., component])**2) / inputs[..., component].nelement()


def optimizer_builder(optimizer, *args, **kwargs):
    return lambda network: optimizer(network.parameters(), *args, **kwargs)


def save_distance_matrix(filename, D, y=None):
    with open(filename, 'wb') as f:
        pickle.dump((D, None), f)


def load_distance_matrix(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def symmetric_part(matrix):
    return 0.5 * (matrix + matrix.transpose())


def antisymmetric_part(matrix):
    return 0.5 * (matrix - matrix.transpose())
