import torch
from torch.nn import Module
from torch.nn.functional import mse_loss
from ..common import batch_determinant
from .utils import single_component_mse, torch_square_grid


class ShapeDistance:
    def __init__(self, q, r, k=32, h=3.4e-4):
        self.q = q
        self.r = r
        self.X = torch_square_grid(k).reshape(-1, 2)
        self.Q = q(self.X)
        self.h = h

    def __call__(self, network):
        Y = network(self.X)
        U = batch_determinant(network.derivative(self.X, self.h))
        loss = mse_loss(self.q(self.X), torch.sqrt(U + 1e-7) * self.r(Y)) * 3.
        self.loss = float(loss)
        return loss

    def get_last(self):
        return self.loss

    def to(self, device):
        self.X = self.X.to(device)
        self.Q = self.Q.to(device)
        return self



class ScaledDistance(ShapeDistance):
    def __init__(self, q, r, k=32, h=3.4e-4):
        self.k = k
        super().__init__(q, r, k, h)

    def __call__(self, network):
        Y = network(self.X)
        U = batch_determinant(network.derivative(self.X, self.h))
        loss = ((self.Q[..., self.component] - torch.sqrt(U) * self.r(Y)[..., self.component]) ** 2 / U).sum() * 2 / self.k**2
        self.loss = float(loss)
        return loss


class SingleComponentScaledLoss(ShapeDistance):
    def __init__(self, q, r, component=2, k=32, h=3.4e-4, **kwargs):
        super().__init__(q, r, k=k, h=h, **kwargs)
        self.component = component
        self.k = k
        self.h = h

    def __call__(self, network):
        Y = network(self.X)
        U = batch_determinant(network.derivative(self.X, self.h))
        loss = ((self.Q - torch.sqrt(U) * self.r(Y)) ** 2 / U).sum() * 2 / self.k**2
        self.loss = loss.item()
        return loss




class SingleComponentLoss(ShapeDistance):
    def __init__(self, q, r, component=2, k=32, h=3.4e-4, **kwargs):
        super().__init__(q, r, k, **kwargs)
        self.component = component
        self.h = h

    def __call__(self, network):
        Y = network(self.X)
        U = batch_determinant(network.derivative(self.X, self.h))

        loss = mse_loss(self.q(self.X)[..., self.component],
                        (torch.sqrt(U + 1e-7) * self.r(Y))[..., self.component])

        self.loss = loss.item()
        return loss
