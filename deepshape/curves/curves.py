import enum
from ..common import central_differences
import torch
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

    def compose_component(self, i, f):
        return lambda x: self.C[i](f(x))

    def compose(self, f: Diffeomorphism):
        """ Composition from right with a map f: R -> R """
        # TODO: Allow genereal dimension curves.
        return Curve((self.compose_component(i, f) for i in range(self.dim)))


class Qmap:
    """ Q-map transformation of curves """
    # TODO: Add literary reference to Q-map

    def __init__(self, curve: Curve):
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
        u = self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)
        return torch.where(
            torch.abs(u) < 1e-7,
            torch.zeros((u.shape[0], self.c.dim)),
            self.c.derivative(
                X, h=h) / torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True))
        )

    def inverse(self, X):
        Q = self(X)
        h = 1. / (X.shape[0] - 1)
        points = Q * Q.norm(dim=-1, keepdim=True)
        return h * points.cumsum(dim=0)

    def compose(self, f):
        return SRVT(self.c.compose(f))

        



""" Below is a couple of example curves and diffeomorphisms for testing the 
reparametrization algorithm."""
