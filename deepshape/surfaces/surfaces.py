import torch
import torch.nn as nn


class Diffeomorphism(nn.Module):
    def __init__(self, component_function_tuple):
        super().__init__()
        self.S = tuple(component_function_tuple)
        
    def forward(self, X):
        out = torch.zeros_like(X)
        out[..., 0] = self.S[0](X)
        out[..., 1] = self.S[1](X)
        return out


class Surface(nn.Module):
    """ Torch-compatible surface class. Constructed from a tuple of functions,
    mapping tensor of dim (..., 2) to R^len(tuple) """
    def __init__(self, component_function_tuple, **kwargs):
        super().__init__()
        self.S = tuple(component_function_tuple)
    
    def forward(self, X):
        return torch.cat([ci(X).unsqueeze(dim=-1) for ci in self.S], dim=-1)
    
    def partial_derivative(self, X, component, h=1e-3):
        H = torch.zeros_like(X)
        H[..., component] = h
        return 0.5 * torch.cat([(ci(X + H) - ci(X - H)).unsqueeze(dim=-1) for ci in self.S], dim=-1) / h
    
    def volume_factor(self, X, h=1e-3):
        return torch.norm(self.normal_vector(X, h), dim=-1, keepdim=True)

    def normal_vector(self, X, h=1e-4):
        dfx = self.partial_derivative(X, 0, h)
        dfy = self.partial_derivative(X, 1, h)
        return torch.cross(dfx, dfy, dim=-1)
    
    def compose(self, f : Diffeomorphism):
        """ Composition from the right, of diffeomorphism class mapping 
        tensors (K, 2) -> (K, 2)"""
        return Surface((
            lambda x: self.S[0](f(x)),
            lambda x: self.S[1](f(x)),
            lambda x: self.S[2](f(x))
        ))
    
    
class Qmap(nn.Module):
    def __init__(self, surface: Surface):
        super().__init__()
        self.s = surface
        
    def forward(self, X, h=1e-3):
        return torch.sqrt(self.s.volume_factor(X, h)) * self.s(X)


class SRNF(nn.Module):
    def __init__(self, surface: Surface):
        super().__init__()
        self.s = surface

    def forward(self, X, h=1e-3):
        n = self.s.normal_vector(X)
        return torch.sqrt(self.s.volume_factor(X, h)) * n / torch.norm(n, dim=-1, keepdim=True)