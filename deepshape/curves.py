import torch
import torch.nn as nn

class Curve(nn.Module):
    """ Define a torch-compatible parametrized curve class, with finite
    difference approximation of derivatives, and composition operator. """
    def __init__(self, component_function_tuple):
        super(Curve, self).__init__()
        self.C = tuple(component_function_tuple)
    
    def forward(self, X):
        return torch.cat([ci(X) for ci in self.C], dim=-1)
    
    def derivative(self, X, h=1e-3):
        """ Finite difference approximation of curve velocity. """
        return 0.5 * torch.cat([ci(X + h) - ci(X - h) for ci in self.C], dim=-1) / h
    
    def compose(self, f):
        """ Composition from right with a map f: R -> R """
        # TODO: Create a Difffeomorphism class, and add type hinting.
        # TODO: Allow genereal dimension curves.
        return Curve((lambda x: self.C[0](f(x)), lambda x: self.C[1](f(x))))


class Qmap(nn.Module):
    """ Q-map transformation of curves """
    # TODO: Add reference to Q-map
    def __init__(self, curve: Curve):
        super(Qmap, self).__init__()
        self.c = curve
        
    def forward(self, X, h=1e-4):
        return torch.sqrt(self.c.derivative(X, h=h).norm(dim=-1, keepdim=True)) * self.c(X)