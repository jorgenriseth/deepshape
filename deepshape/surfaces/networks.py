import torch
import torch.nn as nn
from numpy import pi

from ..common import jacobian
from .layers import DeepShapeLayer

class ReparametrizationNetwork(nn.Module):
    def __init__(self, layerlist):
        super().__init__()
        self.layerlist = layerlist
        for layer in layerlist:
            assert isinstance(layer, DeepShapeLayer), "Layers must inherit DeepShapeLayer"
        
    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x
    
    def derivative(self, x, h=1e-4):
        if h is not None:
            return jacobian(self, x, h)
        
        Df = torch.eye(2, 2, device=x.device)
        for layer in self.layerlist:
            Df = layer.derivative(x, h) @ Df
            x = layer(x)
            
        return Df

    def project(self, **kwargs):
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, DeepShapeLayer):
                    module.project(**kwargs)

    def to(self, device):
        for module in self.modules():
            if isinstance(module, DeepShapeLayer):
                module.to(device)
        
        return self