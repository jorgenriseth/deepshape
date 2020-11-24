import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_plot_data(f, k=32):
    K = k**2
    X = torch.rand(K, 2)
    
    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    
    X = torch.cat((X, Y), dim=1)
    
    Z = f(X).detach().numpy().T
    return Z.reshape(3, k, k)


# Simple plotting function for curves.        
def plot_curve(c, npoints=201):
    X = torch.linspace(0, 1, npoints).unsqueeze(-1)
    C = c(X)
    cx, cy = C[:, 0], C[:, 1]
    
    plt.figure()
    plt.plot(cx, cy)
    plt.show()