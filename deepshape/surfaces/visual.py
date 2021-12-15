import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from ..common.utils import antisymmetric_part, symmetric_part


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


def plot_distance_matrix(D, *args, **kwargs):
    distance = (D + D.min()) / (D.max() - D.min())
    S, A = symmetric_part(D), antisymmetric_part(D)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.matshow(distance, vmin=0, vmax=1, *args, **kwargs)
    ax2.matshow(S, vmin=0, vmax=1, *args, **kwargs)
    ax3.matshow(np.abs(A), vmin=0, vmax=1, *args, **kwargs)
    plt.show()


def plot_surface(f, ax=None, colornorm=None, k=32, camera=(30, -60), **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    
    coloring = get_plot_data(f.volume_factor, k=k).squeeze()
    if colornorm is None:
        colors = coloring / coloring.max()
    else:
        colors = colornorm(coloring)
        
    Z = get_plot_data(f, k=k)
    
    ax.plot_surface(*Z, shade=False, facecolors=cm.jet(colors), rstride=1, cstride=1, **kwargs)
    ax.view_init(*camera)
    return ax


def get_common_colornorm(surfaces, k=128):
    colors = [get_plot_data(fi.volume_factor, k=k).squeeze() for fi in surfaces]
    return Normalize(vmin=min([ci.min() for ci in colors]), vmax=max([ci.max() for ci in colors]))