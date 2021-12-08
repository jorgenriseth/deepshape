import torch
from torch._C import Value
from torch.nn import Module
from torch.nn.functional import mse_loss
from .utils import col_linspace



class ShapeDistance:
    def __init__(self, q, r, k=128):
        # super().__init__(
        self.k = k
        self.X = col_linspace(0, 1, k)
        self.r = r
        self.q = q
        self.Q = q(self.X)

    def __call__(self, network, h=None):
        Y = network(self.X)
        U = network.derivative(self.X, h)

        # Check for negative derivatives. Retry projection, or raise error.
        count = 0
        if U.min() < 0. or torch.isnan(U.min()):
            raise ValueError(
                f"ProjectionError: derivative minimum is {float(U.min())}")

        loss = mse_loss(self.Q, torch.sqrt(U) * self.r(Y)) * 2.
        # loss = ((self.Q - torch.sqrt(U) * self.r(Y))**2).sum() / self.k

        self.loss = float(loss)
        return loss

    def get_last(self):
        return self.loss

    def to(self, device):
        self.X = self.X.to(device)
        return self


class ScaledDistance:
    def __init__(self, q, r, k=128, *, numeric_stabilizer=1e-8):
        self.q = q
        self.r = r
        self.k = k
        self.X = torch.linspace(0, 1, k).reshape(-1, 1)
        self.Q = q(self.X)
        self.numeric_stab = numeric_stabilizer

    def __call__(self, network, h=None):
        Y = network(self.X)
        U = network.derivative(self.X, h)
        U[U < 1e-5] = 1e-5
        # loss = ((self.Q - torch.sqrt(U) * self.r(Y)) ** 2 * U).sum() * 2 / self.k

        loss = ((self.Q - torch.sqrt(U) * self.r(Y)) ** 2 / U).sum() * 2 / self.k
        # loss = mse_loss(self.q(self.X), torch.sqrt(U) * self.r(Y)) * 2.
        self.loss = float(loss)
        return loss

    def get_last(self):
        return self.loss

    def to(self, device):
        self.X = self.X.to(device)
        return self
