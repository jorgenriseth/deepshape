import torch
from numpy import pi, sqrt

from ..common import CurveLayer, col_linspace, torch_clamp


class SineSeries(CurveLayer):
    def __init__(self, N, init_scale=0., p=1):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N+1, dtype=torch.float)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.p = p
        self.project()

    def forward(self, x):
        return x + ((torch.sin(pi * self.nvec * x) / self.nvec) @ self.weights) / pi

    def derivative(self, x, h=None):
        return 1. + torch.cos(pi * self.nvec * x) @ self.weights

    def project(self):
        with torch.no_grad():
            # Possible since lipschitz-vector is 1 everywhere.
            norm = self.weights.norm(1)
            if norm > 1.0 - 1e-6:
                self.weights *= (1 - 1e-6) / norm

    def to(self, device):
        self.nvec = self.nvec.to(device)
        return self


class UnscaledSineSeries(SineSeries):
    # TODO: Consider removal, as this is a worse version of its parent
    # TODO: Found error, might explain poor performance
    def forward(self, x):
        return x + torch.sin(pi * self.nvec * x) @ self.weights

    def derivative(self, x, h=None):
        return 1. + pi * (self.nvec * torch.cos(pi * self.nvec * x)) @ self.weights

    def project(self, p=1):
        with torch.no_grad():
            norm = pi * (self.weights * self.nvec).norm(1)
            if norm > 1.0 - 1e-6:
                self.weights *= (1 - 1e-6) / norm


class PalaisLayer(SineSeries):
    def __init__(self, N, init_scale=0.):
        super().__init__(N)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(2*N, 1, requires_grad=True)
        )
        self.project(p=1)

    def forward(self, x):
        y = torch.cat([
            torch.sin(pi * self.nvec * x) / self.nvec,
            ((torch.cos(2*pi * self.nvec * x) - 1.) / self.nvec) * 0.5
        ], axis=-1) / pi
        return x + y @ self.weights

    def derivative(self, x, h=None):
        u = torch.cat([
            torch.cos(2*pi * self.nvec * x),
            -torch.sin(2*pi * self.nvec * x)
        ], axis=-1)
        return 1. + u @ self.weights


class DerivativeLayer(CurveLayer):
    def __init__(self, N, init_scale=2.0):
        assert N > 1, "N needs to be >= 2"
        super().__init__()
        self.N = N
        self.P = create_projection_matrix(N)
        self.Xnodes = col_linspace(0, 1, N + 1)
        self.Ynodes = torch.zeros((N, 1))
        self.w = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )

    # def g(w):
    #     return torch.tanh(w)

    # def g(w):
    #     return torch.sin(w)

    def g(self, w, a=100.):
        return torch.tanh(w) - torch.tanh(w / a)

    def forward(self, x):
        u = self.P @ self.g(self.w)
        ind = find_interval(x, self.N)
        y = torch.zeros((self.N, 1))
        y[1:] = torch.cumsum(u[:-1], dim=0) / self.N
        return x + (y[ind] + u[ind] * (x - self.Xnodes[ind]))

    def derivative(self, x, h=None):
        u = self.P @ torch.tanh(self.w)
        ind = find_interval(x, self.N)
        return torch.maximum(u[ind], torch.tensor(1e-7))

    def project(self, **kwargs):
        pass


def create_projection_matrix(N):
    P = torch.ones((N, N)) / N
    return (0.5 * N / (N-1)) * (torch.eye(N) - P.fill_diagonal_(1 / N))


def find_interval(x, N):
    return torch_clamp((x * N).squeeze().long(), 0, N-1)


class HatLayer(CurveLayer):
    def __init__(self):
        return NotImplementedError()


class SpectralDenseLayer(CurveLayer):
    def __init__(self):
        return NotImplementedError()


class BumpReluLayer(CurveLayer):
    def __init__(self):
        return NotImplementedError()
