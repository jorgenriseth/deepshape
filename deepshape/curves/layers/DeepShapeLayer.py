from torch.nn import Module
from abc import ABC, abstractmethod

from ...common import central_differences


class DeepShapeLayer(Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
    def derivative(self, x, h=1e-4):
        return central_differences(self, x, h)
    
    @abstractmethod
    def project(self):
        pass