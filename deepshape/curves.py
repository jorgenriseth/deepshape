import torch
import torch.nn as nn

from numpy import pi


class Diffeomorphism(nn.Module):
    def __init__(self, f):
        super(Diffeomorphism, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)
    
    def derivative(self, x, h=5e-4):
        return 0.5 * (self(x+h) - self(x-h)) / h


class Curve(nn.Module):
    """ Define a torch-compatible parametrized curve class, with finite
    difference approximation of derivatives, and composition operator. """
    def __init__(self, component_function_tuple):
        super(Curve, self).__init__()
        self.C = tuple(component_function_tuple)
    
    def forward(self, X):
        return torch.cat([ci(X) for ci in self.C], dim=-1)
    
    def derivative(self, X, h=5e-4):
        """ Finite difference approximation of curve velocity. """
        return 0.5 * torch.cat([ci(X + h) - ci(X - h) for ci in self.C], dim=-1) / h
    
    def compose(self, f: Diffeomorphism):
        """ Composition from right with a map f: R -> R """
        # TODO: Allow genereal dimension curves.
        return Curve((lambda x: self.C[0](f(x)), lambda x: self.C[1](f(x))))

    def subtract(self, c2):
        return Curve((lambda x: self.C[0](x) - c2.C[0](x), lambda x: self.C[1](x) - c2.C[1](x)))


class Qmap(nn.Module):
    """ Q-map transformation of curves """
    # TODO: Add reference to Q-map
    def __init__(self, curve: Curve):
        super(Qmap, self).__init__()
        self.c = curve
        
    def forward(self, X, h=1e-4):
        return torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)) * self.c(X)

class SRVT(nn.Module):
    """ SRVT of curves """
    def __init__(self, curve: Curve):
        super().__init__()
        self.c = curve

    def forward(self, X, h=1e-4):
        return torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)) * self.c.derivative(X, h=h)


""" Below is a couple of example curves adn diffeomorphism for testing the 
reparametrization algorithm."""
Circle = Curve((
    lambda x: torch.cos(2*pi*x),
    lambda x: torch.sin(2*pi*x)
))

QuadDiff = Diffeomorphism(lambda x: 0.9 * x**2 + 0.1 * x)

LogStepDiff = Diffeomorphism(
    lambda x: (0.5 * torch.log(20*x+1) / torch.log(21*torch.ones(1)) 
    + 0.25 * (1 + torch.tanh(20*(x-0.5)) / torch.tanh(21*torch.ones(1))))
)
