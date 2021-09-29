from torch.nn import Module
from abc import ABC, abstractmethod

from ...common import batch_determinant, jacobian

class DeepShapeLayer(Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass
    
    def derivative(self, x, h=1e-4):
        self.Df = jacobian(self, x, h)
        return self.Df
    
    @abstractmethod
    def project(self):
        pass

    def to(self, device):
        super().to(device)