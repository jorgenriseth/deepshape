import torch
from torch.nn import Module
from torch.nn.functional import mse_loss


class ShapeDistance:
    def __init__(self, q, r, k=128, *, numeric_stabilizer=1e-8):
        super().__init__()
        self.q = q
        self.r = r
        self.X = torch.linspace(0, 1, k).reshape(-1, 1)
        self.numeric_stab = numeric_stabilizer

    def __call__(self, network, h=None):
        Y = network(self.X)
        U = network.derivative(self.X, h)
        loss = mse_loss(self.q(self.X), torch.sqrt(U) * self.r(Y)) * 2.
        self.loss = float(loss)
        return loss

    def get_last(self):
        return self.loss

    def to(self, device):
        self.X = self.X.to(device)
        return self


class ScaledDistance:
    def __init__(self, q, r, k=128, *, numeric_stabilizer=1e-8):
        super().__init__()
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