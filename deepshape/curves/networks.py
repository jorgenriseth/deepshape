import time
import torch
import torch.nn as nn
import numpy as np

# from .layers import FourierLayer
from ..common import central_differences, numpy_nans
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
            return central_differences(self, x, h)
        
        dc = 1.
        for layer in self.layerlist:
            dc = dc *  layer.derivative(x)
            x = layer(x)
        
        return dc

    def project(self, **kwargs):
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, DeepShapeLayer):
                    module.project(**kwargs)


    def to(self, device):
        super().to(device)
        for module in self.modules():
            if isinstance(module, DeepShapeLayer):
                module.to(device)
        
        return self
