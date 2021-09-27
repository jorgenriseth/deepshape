import torch
from torch.nn import Module
from torch.nn.functional import mse_loss

class ShapeDistance(Module):
    def __init__(self, q, r, k=128, *, numeric_stabilizer = 1e-8):
        super().__init__()
        self.q = q
        self.r = r
        self.X = torch.linspace(0, 1, k).reshape(-1, 1)      
        self.numeric_stab = numeric_stabilizer
             
    def forward(self, network, h=1e-4):
        Y = network(self.X)
        U = network.derivative(self.X, h)
        loss = mse_loss(self.q(self.X), torch.sqrt(U + self.numeric_stab) * self.r(Y)) / 2.
        self.loss = loss.item()
        return loss
        
    def get_last(self):
        return self.loss