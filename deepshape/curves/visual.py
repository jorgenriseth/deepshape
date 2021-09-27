import torch
import matplotlib.pyplot as plt

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


def get_plot_data(q, r, network, npoints):
    x = torch.linspace(0, 1, npoints).unsqueeze(-1)
    with torch.no_grad():
        y = network(x)
        u = network.derivative(x)
        Q, R = q(x), torch.sqrt(u) * r(y)
    return x, y, u, Q, R