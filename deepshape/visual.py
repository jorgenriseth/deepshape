import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def linspace(npoints):
    return torch.linspace(0, 1, npoints).unsqueeze(-1)

def get_plot_data(f, k=32):
    K = k**2
    X = torch.rand(K, 2)
    
    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    
    X = torch.cat((X, Y), dim=1)
    
    Z = f(X).detach().numpy().T
    return Z.reshape(-1, k, k)

def get_plot_data1D(q, r, network, npoints):
    x = torch.linspace(0, 1, npoints).unsqueeze(-1)

    z, y = network(x)
    z, y = z.detach(), y.detach()
    Q, R = q(x), network.reparametrized(r, x)
    R = R.detach()
    return z, y, Q, R


# Simple plotting function for curves.        
def plot_curve(c, npoints=201, dotpoints=None, ax=None, **kwargs):
    X = torch.linspace(0, 1, npoints).unsqueeze(-1)
    C = c(X)
    cx, cy = C[:, 0], C[:, 1]
    
    if ax is None:
        fig, ax = plt.subplots()
         
    ax.plot(cx, cy, **kwargs)

    if dotpoints is not None:
        X = torch.linspace(0, 1, dotpoints).unsqueeze(-1)
        C = c(X)
        cx, cy = C[:, 0], C[:, 1]
        ax.plot(cx, cy, c=ax.lines[-1].get_color(), ls='none', marker='o')