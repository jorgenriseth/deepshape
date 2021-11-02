import torch
import numpy as np


def load_curve_data(filename, columns=None):
    # Load array from file, and convert to torch tensor
    X = torch.tensor(np.loadtxt(filename, skiprows=1, usecols=columns), dtype=torch.float32)
    # Ensure column vectors
    if X.dim() == 1:
        return X.unsqueeze(-1)
    return X


def col_linspace(start, end, N):
    return torch.linspace(start, end, N).unsqueeze(-1)


def get_function_data(filename, column):
    y = load_curve_data(filename, column)
    X = col_linspace(0, 1, y.size()[0])
    return X, y
