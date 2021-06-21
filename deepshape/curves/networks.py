import time
import torch
import torch.nn as nn
import numpy as np

from .layers import FourierLayer

class ReparametrizationNetwork(nn.Module):
    def __init__(self, L, N, init_scale=0.0, layer_type=FourierLayer):
        # Convert N to list of length N (if not already there)
        if type(N) == int:
            N = [N for _ in range(L)]
        assert len(N) == L, "N should be of length L (or a single number) "
        
        # Init list of layers
        super().__init__()
        self.layers = nn.ModuleList([layer_type(N[l], init_scale=init_scale) for l in range(L)])
        self.gotnan = False
        
    def forward(self, X):
        Z, Y = X, torch.ones_like(X)
        for l in self.layers:
            Z, Y0 = l(Z)
            Y *= Y0
        return Z, Y
    
    def reparametrized(self, r, X):
        Z, Y = self(X)
        return torch.sqrt(Y) * r(Z)



def train(q, r, network, optimizer, scheduler=None, loss=nn.MSELoss(), 
          npoints=1024, iterations=300, epsilon=None, log_every=10):
    """ General purpose function for training a curve reparametrization network,
    which works with most optimizers not requiring a closure.

    TODO: Implement as method of the network.
    """
    tic = time.time()
    
    # Initialize node placement
    x = torch.linspace(0, 1, npoints).unsqueeze(-1)
    
    # Evaluate initial error
    error = np.empty(iterations+1)
    error.fill(np.nan)
    Z, Y = network(x)
    Q = q(x)
    R = network.reparametrized(r, x)
    error[0] = loss(R, Q) * 2


    for i in range(iterations):   
        x = torch.linspace(0, 1, npoints).unsqueeze(-1)
        
        # Set gradient buffers to zero.
        optimizer.zero_grad()

        # Find current reparametrized Q-maps
        Z, Y = network(x)
        Q = q(x)
        R = network.reparametrized(r, x)

        # Compute loss, and perform a backward pass and gradient step
        l = loss(R, Q) * 2
        l.backward()

        if scheduler is not None:
            scheduler.step(l)
        
        optimizer.step()
        error[i+1] = l.item()

        # Projection step
        with torch.no_grad():
            for layer in network.layers:
                layer.project(npoints, epsilon=epsilon)

        if log_every > 0 and i % log_every == 0:
            print('[Iter %5d] loss: %.5f' %
                  (i + 1, l))        

    toc = time.time()

    print()
    print(f'Finished training in {toc - tic:.5f}s')
    return error



def train_bfgs(q, r, network, optimizer, iterations=5, 
        scheduler=None, loss=nn.MSELoss(), npoints=1024, epsilon=None, log_every=10):
    """ Function for training a reparametrization network using the LBFGS 
    optimizer """
    tic = time.time()

    # Get max iterations from optimizer
    bfgs_max_iter = optimizer.defaults["max_iter"]
    
    # Initialize node placement
    x = torch.linspace(0, 1, npoints).unsqueeze(-1)
    
    # Evaluate initial error
    error = np.empty(bfgs_max_iter * (iterations+1))
    error.fill(np.nan)
    Z, Y = network(x)
    Q = q(x)
    R = network.reparametrized(r, x)
    error[0] = loss(R, Q) * 2

    for i in range(iterations):
        inner_count = [0]

        def closure():
            # Set gradient buffers to zero.
            optimizer.zero_grad()
            
            # Projection step
            with torch.no_grad():
                for layer in network.layers:
                    # if torch.isnan(layer.weights).any():
                    #     print("Weights broken by updates.")
                    layer.project(2 * npoints, epsilon=epsilon)

            # Find current reparametrized Q-maps
            Q = q(x)
            R = network.reparametrized(r, x)

            if torch.isnan(R).any():
                print("Warning: Got NaN. Returning.")
                network.gotnan = True
                l = loss(R, Q) * 0.
                l.backward()
                return l

            # Compute loss, and perform a backward pass and gradient step
            l = loss(R, Q) * 2
            l.backward()

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step(l)

            # Save Loss Function
            j = i * bfgs_max_iter + inner_count[0]
            error[j+1] = l.item()
            inner_count[0] += 1

            if log_every > 0 and j % log_every == 0:
                print('[Iter %5d] loss: %.5f' % (j + 1, l))
            return l

        optimizer.step(closure)
        
        if network.gotnan:
            break

    toc = time.time()

    print()
    print(f'Finished training in {toc - tic:.5f}s')
    j = i * bfgs_max_iter + inner_count[0]
    return error[:j+1]
