from .utils import col_linspace
from torch.nn.functional import mse_loss
from torch.nn import Module
from torch._C import Value
from ..common import central_differences
import torch
import torch.nn as nn
import numpy as np
from numpy import pi


class Diffeomorphism:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def derivative(self, x, h=5e-4):
        return central_differences(self, x, h)

    def compose(self, f):
        return Diffeomorphism(lambda x: self(f(x)))


class Curve:
    """ Define a torch-compatible parametrized curve class, with finite
    difference approximation of derivatives, and composition operator. """

    def __init__(self, component_function_tuple):
        self.C = tuple(component_function_tuple)
        self.dim = len(self.C)

    def __call__(self, X):
        return torch.cat([ci(X) for ci in self.C], dim=-1)

    def derivative(self, X, h=1e-4):
        """ Finite difference approximation of curve velocity. """
        return torch.cat([central_differences(ci, X, h) for ci in self.C], dim=-1)

    def compose(self, f: Diffeomorphism):
        """ Composition from right with a map f: R -> R """
        # TODO: Allow genereal dimension curves.
        return Curve((lambda x: ci(f(x)) for ci in self.C))

    def subtract(self, c2):
        return Curve((lambda x: self.C[i](x) - c2.C[i](x) for i in range(self.dim)))


class Qmap:
    """ Q-map transformation of curves """
    # TODO: Add literary reference to Q-map

    def __init__(self, curve: Curve):
        super().__init__()
        self.c = curve

    def __call__(self, X, h=1e-4):
        return torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)) * self.c(X)


class SRVT:
    """ SRVT of curves """
    # TODO: Add literary reference to SRVT

    def __init__(self, curve: Curve):
        super().__init__()
        self.c = curve

    def __call__(self, X, h=1e-4):
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
        )) / np.sqrt(3.)


class HalfCircle(Curve):
    def __init__(self, transform="qmap"):
        if transform.lower() == "qmap":
            super().__init__((
                lambda x: torch.cos(pi * x) / np.cbrt(pi),
                lambda x: torch.sin(pi * x) / np.cbrt(pi)
            ))
        elif transform.lower() == "srvt":
            super().__init__((
                lambda x: torch.sin(pi * x) / pi,
                lambda x: -torch.cos(pi * x) / pi
            ))
        else:
            raise ValueError("invalid transform. Must be 'srvt' or 'qmap'")


class Line(Curve):
    def __init__(self, transform=""):
        if transform.lower() == "qmap":
            super().__init__((
                lambda x: torch.zeros_like(x),
                lambda x: torch.pow(3*x + 1., (1./3.))
            ))
        elif transform.lower() == "srvt":
            super().__init__((
                lambda x: torch.ones_like(x),
                lambda x: x - 0.5
            ))
        else:
            raise ValueError("invalid transform. Must be 'srvt' or 'qmap'")


class QuadDiff(Diffeomorphism):
    def __init__(self):
        super().__init__(lambda x: 0.9 * x**2 + 0.1 * x)


class LogStepDiff(Diffeomorphism):
    def __init__(self, a=20):
        super().__init__(
            lambda x: (0.5 * torch.log(a*x+1) / np.log(a+1.)
                       + 0.25 * (1 + torch.tanh(a*(x-0.5)) / np.tanh(a+1.)))
        )


class OptimalCircleLine(Diffeomorphism):
    def __init__(self):
        super().__init__(lambda x: x - 0.5 * torch.sin(2*pi*x) / pi)
