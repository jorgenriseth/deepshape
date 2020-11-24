import time
import torch
import torch.nn as nn

from layers import FourierLayer1D, FourierLayer2D

class ReparametrizationNetwork2D(nn.Module):
    def __init__(self, L, N):
        # Convert N to list of length N (if not already there)
        if type(N) == int:
            N = [N for _ in range(L)]
        assert len(N) == L, "N should be of length L (or a single number) "
        
        # Init list of layers
        super(ReparametrizationNetwork2D, self).__init__()
        self.layers = nn.ModuleList([FourierLayer2D(N[l]) for l in range(L)])
        
    def forward(self, X):
        Z, J = X, 1.
        for l in self.layers:
            Z, J0 = l(Z)
            J = J0 * J
        return Z, J
    
    def reparametrized(self, r, X):
        Z, J = self(X)
        return torch.sqrt(J.view(-1, 1)) * r(Z)


    def train(self, q, r, optimizer, loss=nn.MSELoss(), nxpoints=32,
        iterations=300):

        # Create Datapoints
        # TODO: Create Dataloader
        k = nxpoints
        K = k**2
        X = torch.rand(K, 2)
        X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
        X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
        X = torch.cat((X, Y), dim=1)
        
        for i in range(iterations):    
            # Set gradient buffers to zero.
            optimizer.zero_grad()
            
            # Find current reparametrized Q-maps
            # Z, J = RN(X)  # Forward Pass
            Q = q(X)
            R = self.reparametrized(r, X)
            
            # Compute loss, and perform a backward pass and gradient step
            l = loss(Q, R)
            l.backward()
            optimizer.step()

            if i % 10 == 0:
                print('[Iter %5d] loss: %.5f' %
                    (i + 1, l))
                
            # Should insert projection step here as well (has not been necessary until now) 
        toc = time.time()


class ReparametrizationNetwork1D(nn.Module):
    def __init__(self, L, N):
        # Convert N to list of length N (if not already there)
        if type(N) == int:
            N = [N for _ in range(L)]
        assert len(N) == L, "N should be of length L (or a single number) "
        
        # Init list of layers
        super(ReparametrizationNetwork1D, self).__init__()
        self.layers = nn.ModuleList([FourierLayer1D(N[l]) for l in range(L)])
        
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
         iterations=300):
    tic = time.time()
    
    # Initialize node placement
    x = torch.linspace(0, 1, npoints).unsqueeze(-1)
    for i in range(iterations):    
        # Set gradient buffers to zero.
        optimizer.zero_grad()

        # Find current reparametrized Q-maps
        Z, Y = network(x)
        Q = q(x)
        R = network.reparametrized(r, x)

        # Compute loss, and perform a backward pass and gradient step
        l = loss(Q, R)
        l.backward()
        optimizer.step()

        with torch.no_grad():
            for layer in network.layers:
                layer.project(npoints)

        if i % 10 == 0:
            print('[Iter %5d] loss: %.5f' %
                  (i + 1, l))

        # Should insert projection step here as well (has not been necessary until now)

    toc = time.time()

    print()
    print(f'Finished training in {toc - tic:.5f}s')