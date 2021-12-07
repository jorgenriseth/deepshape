import torch
import torch.nn as nn
from .DeepShapeLayer import DeepShapeLayer
from torch.nn import Linear, Tanh, ReLU


def power_iteration(W: torch.Tensor, iters: int = 5):
    v = torch.rand(W.size(1))
    vnorm = v.norm()

    for i in range(iters):
        v = torch.mv( torch.mm(W.transpose(1, 0), W), v / vnorm )
        vnorm = v.norm()
        
    return (W @ v).norm() / vnorm


def spectral_normalization(W : torch.Tensor, c : float, iters: int = 5):
    spec_norm = torch.linalg.norm(W, 2)
    W *= min(c, c / spec_norm)
    return W


class SpectralDenseLayer(DeepShapeLayer):
    def __init__(self, input_dim : int, output_dim: int, bias : bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = Linear(input_dim, output_dim, bias=bias)
        self.project(0.9)

    def forward(self, x):
        return torch.leakyrelu(self.linear(x))

    def project(self, c : float = 0.9):
        with torch.no_grad():
            # k = float(torch.linalg.norm(self.linear.weight, 2))
            # if k >= 1:
            #     self.linear.weight *= c / k
            spectral_normalization(self.linear.weight, c)


class InvertibleResidualBlock(DeepShapeLayer):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # self.lin1 = torch.nn.Parameter()
        self.lin1 = Linear(1, hidden_dim, bias=True)
        self.lin2 = Linear(hidden_dim, 1, bias=False)
        with torch.no_grad():
            self.lin1.bias.data = torch.linspace(-2., 2., hidden_dim)
            self.lin1.weight.data = (torch.rand(hidden_dim) - 2 * self.lin1.bias.data).view(hidden_dim, 1)
            # self.lin2.weight.data = torch.zeros(1, hidden_dim)

        # self.lin1.weight = torch.

        # with torch.no_grad():
        #     self.lin1.weight *= 0.
        #     self.lin2.weight *= 0.
        self.project(1.)

    def unnormalized(self, x):
        return x + self.lin2(torch.relu(self.lin1(x)))

    def forward(self, x):
        return self.unit_interval(self.unnormalized(x))

    def unit_interval(self, x):
        x0 = self.unnormalized(torch.zeros(1))
        x1 = self.unnormalized(torch.ones(1))
        return (x - x0) / (x1 - x0)

    def project(self, c : float = 0.9):
        with torch.no_grad():
            spectral_normalization(self.lin1.weight, c) 
            spectral_normalization(self.lin2.weight, c)

