import torch
from torch.nn import Module
from torch.nn.functional import mse_loss
from ..common import batch_determinant
from .utils import single_component_mse, torch_square_grid

class ShapeDistance:
    def __init__(self, q, r, k=32):
        self.q = q
        self.r = r
        self.X = torch_square_grid(k).reshape(-1, 2)
             
    def __call__(self, network, h=3.4e-4):
        Y = network(self.X)
        U = batch_determinant(network.derivative(self.X, h))
        loss = mse_loss(self.q(self.X), torch.sqrt(U) * self.r(Y)) * 3.
        self.loss = float(loss)
        return loss
        
    def get_last(self):
        return self.loss

    def to(self, device):
        self.X = self.X.to(device)
        return self


class SingleComponentLoss(ShapeDistance):
    def __init__(self, q, r, k=32, component=2, **kwargs):
        super().__init__(q, r, k, **kwargs)
        self.component = component
             
    def forward(self, network, h=None):
        Y = network(self.X)
        U = batch_determinant(network.derivative(self.X, h))
        loss = single_component_mse(self.q(self.X), torch.sqrt(U) * self.r(Y),
            self.component)

        self.loss = loss.item()
        return loss
