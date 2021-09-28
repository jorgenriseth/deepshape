import torch
from torch.nn import Module
from torch.nn.functional import mse_loss
from ..common import batch_determinant
from .utils import torch_square_grid

class ShapeDistance(Module):
    def __init__(self, q, r, k=32, *, numeric_stabilizer = 1e-7):
        super().__init__()
        self.q = q
        self.r = r
        self.X = torch_square_grid(k).reshape(-1, 2)
        self.numeric_stab = numeric_stabilizer
             
    def forward(self, network, h=5e-4):
        Y = network(self.X)
        U = batch_determinant(network.derivative(self.X, h))
        loss = mse_loss(self.q(self.X), torch.sqrt(U) * self.r(Y)) * 3.
        self.loss = loss.item()
        return loss
        
    def get_last(self):
        return self.loss