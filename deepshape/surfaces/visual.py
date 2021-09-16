import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

from .utils import symmetrize_matrix


def get_plot_data(f, k=32):
    K = k**2
    X = torch.rand(K, 2)
    
    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    
    X = torch.cat((X, Y), dim=1)
    
    Z = f(X).detach().numpy().T
    return Z.reshape(-1, k, k)


def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    
    
def plot_diffeomorphism(f, k=16, ax=None, **kwargs):
    K = k**2
    X = torch.rand(K, 2)
    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    X = torch.cat((X, Y), dim=1)
    Z = f(X).reshape(k, k, 2).detach()
    plot_grid(Z[:, :, 0], Z[:, :, 1], ax=ax, **kwargs)


def plot_clustering(X, labels, cluster, title=None, figsize=(8, 6)):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X_norm = (X - x_min) / (x_max - x_min) 

    # Create figure. 
    # TODO: Reconfigure to use optional axis.
    plt.figure(figsize=figsize)
    for i in range(X.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]),
                 color=plt.cm.nipy_spectral(cluster[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # Clear ticks
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    if title is not None:
        plt.title(title)


def plot_distance_matrix(D):
    distance = (D + D.min()) / (D.max() - D.min())
    S, A = symmetrize_matrix(D)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.matshow(distance, vmin=0, vmax=1)
    ax2.matshow(S, vmin=0, vmax=1)
    ax3.matshow(np.abs(A), vmin=0, vmax=1)
    plt.show()
