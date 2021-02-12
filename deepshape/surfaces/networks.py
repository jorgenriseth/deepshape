import time
import torch
import torch.nn as nn
import numpy as np

from surfaces.layers import FourierLayer


class ReparametrizationNetwork2D(nn.Module):
    def __init__(self, L, N, init_scale=0.0, layer_type=FourierLayer):
        # Convert N to list of length N (if not already there)
        if type(N) == int:
            N = [N for _ in range(L)]
        assert len(N) == L, "N should be of length L (or a single number) "
        
        # Init list of layers
        super().__init__()
        self.layers = nn.ModuleList([layer_type(N[l], init_scale=init_scale) for l in range(L)])
        
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
        """ General training method for for surface reparametrization
        with most pytorch optimizers not requiiring closures."""
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


def train_bfgs(q, r, network, optimizer, iterations=1, 
        scheduler=None, loss=nn.MSELoss(), npoints=32, epsilon=None, log_every=10):
    """ Function for training a reparametrization network for surfaces using the LBFGS 
    optimizer """
    tic = time.time()

    # Get max iterations from optimizer
    max_iter = optimizer.defaults["max_iter"]
    
    # Create Datapoints
    k = npoints
    K = npoints**2
    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
    X = torch.cat((X, Y), dim=1)
    
    # Evaluate initial error
    errors = np.empty(max_iter * (iterations + 1))
    errors[:] = np.nan
    Z, Y = network(X)
    Q = q(X)
    R = network.reparametrized(r, X)
    errors[0] = loss(R, Q) * 3  # Multiply by 3 (dimension of points)

    # Start Training 
    for i in range(iterations):
        print("Iteration i")
        inner = [0]
        
        def closure():
            # Set gradient buffers to` zero.
            optimizer.zero_grad()
            
            with torch.no_grad():
                Z = X
                for layer in network.layers:
                    layer.project(Z, 1e-3, 1e-3)
                    Z, _ = layer(Z)

            Q = q(X)
            R = network.reparametrized(r, X)

            # Compute loss, and perform a backward pass and gradient step
            l = loss(Q, R) * 3
            l.backward()

            if scheduler:
                scheduler.step(l)
            
            j = i * max_iter + inner[0]
            errors[j] = l.item()
            inner[0] += 1

            
            print('[Iter %5d] loss: %.8f' % (j, l))
            return l
        
        optimizer.step(closure)
                
    toc = time.time()
    print()
    print(f'Finished training in {toc - tic:.5f}s')

    j = i * max_iter + inner[0]
    return errors[:j]
