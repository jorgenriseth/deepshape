from torch.nn import Module
from abc import ABC, abstractmethod

from ...common import batch_determinant, jacobian


class DeepShapeLayer(Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
    def derivative(self, x, h=1e-4):
        return jacobian(self, x, h)
    
    @abstractmethod
    def project(self):
        pass