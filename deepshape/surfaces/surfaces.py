import torch
import torch.nn as nn
import numpy as np


class Diffeomorphism:
    def __init__(self, component_function_tuple):
        super().__init__()
        self.S = tuple(component_function_tuple)

    def __call__(self, X):
        out = torch.zeros_like(X)
        out[..., 0] = self.S[0](X)
        out[..., 1] = self.S[1](X)
        return out

    def compose(self, f: 'Diffeomorphism'):
        """ Composition from the right, of diffeomorphism class mapping 
        tensors (K, 2) -> (K, 2)"""
        return Diffeomorphism((
            lambda x: self.S[0](f(x)),
            lambda x: self.S[1](f(x)),
        ))


class Surface:
    """ Torch-compatible surface class. Constructed from a tuple of functions,
    mapping tensor of dim (..., 2) to R^len(tuple) """

    def __init__(self, component_function_tuple, **kwargs):
        super().__init__(**kwargs)
        self.S = tuple(component_function_tuple)
        self.dim = len(self.S)

    def __call__(self, X):
        return torch.cat([ci(X).unsqueeze(dim=-1) for ci in self.S], dim=-1)

    def partial_derivative(self, X, component, h=3.4e-4):
        H = torch.zeros_like(X)
        H[..., component] = h
        return 0.5 * torch.cat([(ci(X + H) - ci(X - H)).unsqueeze(dim=-1) for ci in self.S], dim=-1) / h

    def volume_factor(self, X, h=1e-4):
        return torch.norm(self.normal_vector(X, h), dim=-1, keepdim=True)

    def normal_vector(self, X, h=3.4e-4):
        dfx = self.partial_derivative(X, 0, h)
        dfy = self.partial_derivative(X, 1, h)
        return torch.cross(dfx, dfy, dim=-1)

    def compose(self, f: Diffeomorphism):
        """ Composition from the right, of diffeomorphism class mapping 
        tensors (K, 2) -> (K, 2)"""
        return Surface((
            lambda x: self.S[0](f(x)),
            lambda x: self.S[1](f(x)),
            lambda x: self.S[2](f(x))
        ))


class Qmap:
    def __init__(self, surface: Surface):
        self.s = surface

    def __call__(self, X, h=1e-3):
        return torch.sqrt(self.s.volume_factor(X, h)) * self.s(X)


class SRNF:
    def __init__(self, surface: Surface):
        self.s = surface

    def __call__(self, X, h=1e-3):
        n = self.s.normal_vector(X)
        u = torch.norm(n, dim=-1, keepdim=True)
        return torch.where(
            torch.abs(u) < 1e-7,
            torch.zeros((u.shape[0], self.s.dim)),
            torch.sqrt(self.s.volume_factor(X, h)) * n / torch.norm(n, dim=-1, keepdim=True)
        )

# Examples surfaces and warps
class CylinderWrap(Surface):
    def __init__(self):
        super().__init__((
            lambda x: torch.sin(2*np.pi*x[..., 0]),
            lambda x: torch.sin(4*np.pi*x[..., 0]),
            lambda x: x[..., 1]
        ))


class HyperbolicParaboloid(Surface):
    def __init__(self):
        super().__init__((
            lambda x: x[..., 0],
            lambda x: x[..., 1],
            lambda x: (x[..., 0] - 0.5)**2 - (x[..., 1] - 0.5)**2
        ))


class LogStepQuadratic(Diffeomorphism):
    def __init__(self, a=20., b=0.1):
        assert 0. < b < 1., "b must be between 0 and 1."
        c = (1 - b)
        super().__init__((
            lambda x: c * x[..., 0]**2 + b * x[..., 0],
            lambda x: (0.5 * torch.log(a*x[..., 1]+1) / torch.log(21*torch.ones(1))
                       + 0.25 * (1 + torch.tanh(a*(x[..., 1]-0.5)) / torch.tanh(21*torch.ones(1))))
        ))


# Helper function to create Rotation-Diffeomorphism
def angle(x):
    return 0.5 * np.pi * torch.sin(np.pi * x[..., 0]) * torch.sin(np.pi * x[..., 1])


class RotationDiffeomorphism(Diffeomorphism):
    def __init__(self):
        super().__init__((
            lambda x: (x[..., 0] - 0.5) * torch.cos(angle(x)) -
            (x[..., 1] - 0.5) * torch.sin(angle(x)) + 0.5,
            lambda x: (x[..., 0] - 0.5) * torch.sin(angle(x)) +
            (x[..., 1] - 0.5) * torch.cos(angle(x)) + 0.5
        ))
