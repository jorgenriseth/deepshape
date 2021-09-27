from ..common import central_differences
import torch
import torch.nn as nn

from numpy import pi


class Diffeomorphism(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)
    
    def derivative(self, x, h=5e-4):
        return central_differences(self, x, h)


class Curve(nn.Module):
    """ Define a torch-compatible parametrized curve class, with finite
    difference approximation of derivatives, and composition operator. """
    def __init__(self, component_function_tuple):
        super().__init__()
        self.C = tuple(component_function_tuple)
    
    def forward(self, X):
        return torch.cat([ci(X) for ci in self.C], dim=-1)
    
    def derivative(self, X, h=1e-4):
        """ Finite difference approximation of curve velocity. """
        return torch.cat([central_differences(ci, X, h) for ci in self.C], dim=-1)
    
    def compose(self, f: Diffeomorphism):
        """ Composition from right with a map f: R -> R """
        # TODO: Allow genereal dimension curves.
        return Curve((lambda x: self.C[0](f(x)), lambda x: self.C[1](f(x))))

    def subtract(self, c2):
        return Curve((lambda x: self.C[0](x) - c2.C[0](x), lambda x: self.C[1](x) - c2.C[1](x)))


class Qmap(nn.Module):
    """ Q-map transformation of curves """
    # TODO: Add literary reference to Q-map
    def __init__(self, curve: Curve):
        super().__init__()
        self.c = curve
        
    def forward(self, X, h=1e-4):
        return torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)) * self.c(X)


class SRVT(nn.Module):
    """ SRVT of curves """
    def __init__(self, curve: Curve):
        super().__init__()
        self.c = curve

    def forward(self, X, h=1e-4):
        return self.c.derivative(X, h=h) / torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True))


""" Below is a couple of example curves and diffeomorphisms for testing the 
reparametrization algorithm."""
class Circle(Curve):
    def __init__(self):
        super().__init__((
            lambda x: torch.cos(2*pi*x),
            lambda x: torch.sin(2*pi*x)
        ))


class Infinity(Curve):
    def __init__(self):
        super().__init__((
            lambda x: torch.cos(2*pi*x),
            lambda x: torch.sin(4*pi*x)
        ))


class QuadDiff(Diffeomorphism):
    def __init__(self):
        super().__init__(lambda x: 0.9 * x**2 + 0.1 * x)


class LogStepDiff(Diffeomorphism):
    def __init__(self):
        super().__init__(
            lambda x: (0.5 * torch.log(20*x+1) / torch.log(21*torch.ones(1)) 
                + 0.25 * (1 + torch.tanh(20*(x-0.5)) / torch.tanh(21*torch.ones(1))))
        )
