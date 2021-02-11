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

    z, y = network(x)
    z, y = z.detach(), y.detach()
    Q, R = q(x), network.reparametrized(r, x)
    R = R.detach()
    return x, z, y, Q, R