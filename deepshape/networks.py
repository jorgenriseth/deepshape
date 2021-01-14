import time
import torch
import torch.nn as nn
import numpy as np

from layers import FourierLayer1D, FourierLayer2D

class ReparametrizationNetwork2D(nn.Module):
    def __init__(self, L, N, init_scale=0.0):
        # Convert N to list of length N (if not already there)
        if type(N) == int:
            N = [N for _ in range(L)]
        assert len(N) == L, "N should be of length L (or a single number) "
        
        # Init list of layers
        super().__init__()
        self.layers = nn.ModuleList([FourierLayer2D(N[l], init_scale=init_scale) for l in range(L)])
        
    def forward(self, X):
        Z, J = X, 1.
        for l in self.layers:
            Z, J0 = l(Z)
            J = J0 * J
        return Z, J
    
    def reparametrized(self, r, X):
        Z, J = self(X)
        return torch.sqrt(J) * r(Z)


    def train(self, q, r, optimizer, loss=nn.MSELoss(), nxpoints=32,
        iterations=300, printiter=20, epsilon=5e-1, delta=1e-6):
        tic = time.time()

        # Create Datapoints
        # TODO: Create Dataloader
        k = nxpoints
        K = k**2
        X = torch.rand(K, 2)
        X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
        X = torch.cat((X, Y), dim=1)

            # Evaluate initial error
        error = np.empty(iterations+1)
        error.fill(np.nan)
        Z, _ = self(X)
        Q = q(X)
        R = self.reparametrized(r, X)
        error[0] = loss(R, Q)
        
        for i in range(iterations):    
            # Set gradient buffers to zero.
            optimizer.zero_grad()
            
            Q = q(X)
            R = self.reparametrized(r, X)
            
            # Compute loss, and perform a backward pass and gradient step
            l = loss(Q, R)
            l.backward()
            optimizer.step()
            error[i+1] = l.item()

            with torch.no_grad():
                Z = X
                for layer in self.layers:
                    layer.project(Z, epsilon, delta)
                    Z, _ = layer(Z)


            if i % printiter == 0:
                print('[Iter %5d] loss: %.5f' %
                    (i + 1, l))

                        # Find current reparametrized Q-maps
            Z, J = self(X)  # Forward Pass
            if J.min().item() < 0.:
                ind = J.argmin().item()
                print("Iter", i, "   J min: ", J.min().item(), J.argmin())

                
            # Should insert projection step here as well (has not been necessary until now) 
        toc = time.time()
        print()
        print(f'Finished training in {toc - tic:.5f}s')
        return error


class ReparametrizationNetwork1D(nn.Module):
    def __init__(self, L, N, init_scale=0.0):
        # Convert N to list of length N (if not already there)
        if type(N) == int:
            N = [N for _ in range(L)]
        assert len(N) == L, "N should be of length L (or a single number) "
        
        # Init list of layers
        super().__init__()
        self.layers = nn.ModuleList([FourierLayer1D(N[l], init_scale=init_scale) for l in range(L)])
        
    def forward(self, X):
        Z, Y = X, torch.ones_like(X)
        for l in self.layers:
            Z, Y0 = l(Z)
            Y *= Y0
        return Z, Y
    
    def reparametrized(self, r, X):
        Z, Y = self(X)
        return torch.sqrt(Y) * r(Z)



def train(q, r, network, optimizer, loss=nn.MSELoss(), npoints=1024,
         iterations=300, epsilon=None, log_every=10):
    tic = time.time()
    
    # Initialize node placement
    x = torch.linspace(0, 1, npoints).unsqueeze(-1)
    
    # Evaluate initial error
    error = np.empty(iterations+1)
    error.fill(np.nan)
    Z, Y = network(x)
    Q = q(x)
    R = network.reparametrized(r, x)
    error[0] = loss(R, Q)


    for i in range(iterations):   
        x = torch.linspace(0, 1, npoints).unsqueeze(-1)

        # Set gradient buffers to zero.
        optimizer.zero_grad()

        # Find current reparametrized Q-maps
        Z, Y = network(x)
        Q = q(x)
        R = network.reparametrized(r, x)

        # Compute loss, and perform a backward pass and gradient step
        l = loss(R, Q)
        l.backward()
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